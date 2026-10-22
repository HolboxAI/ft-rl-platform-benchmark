#!/usr/bin/env python3
"""Hardware-agnostic GRPO trainer (plain PyTorch + transformers).

One file, one algorithm, five backends (cpu / cuda / neuron / tpu / tpu-xla).
The GRPO math follows the verl / Miles configuration used in
`ft-rl-platform-benchmark-main 2`:

  * group-normalised advantages:  A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
  * clipped policy-gradient loss with asymmetric clip (0.2 / 0.28, Miles defaults)
  * KL penalty in the loss, `low_var_kl` (k3) estimator, coefficient --kl-beta
  * token-mean loss aggregation over the whole rollout batch
  * one optimizer step per rollout batch (rollout_batch_size x n == global batch),
    so the run is fully on-policy and the importance ratio is exactly 1.

Run from the repo root:

    python -m common.grpo.train_grpo --backend cpu \
        --model ~/models/cholec-sft-qwen3.5-4b --num-steps 1 --out-dir /tmp/smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform as _platform
import statistics
import sys
import time
from datetime import datetime, timezone

import torch

from common.grpo.device import BACKENDS, get_backend
from common.grpo.rollout import generate_hf, generate_static, postprocess, response_logprobs
from common.grpo.tasks import make_task, to_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--platform", default=None, help="platform label for final_results.json (default: backend)")
    p.add_argument("--task", choices=("cholect50",), default="cholect50")
    p.add_argument("--model", default=os.path.expanduser("~/models/cholec-sft-qwen3.5-4b"),
                   help="merged CholecT50 SFT checkpoint (s3://stanford-train-data/models/cholec-sft-qwen3.5-4b)")
    p.add_argument("--data", default=None, help="override the verl parquet dir (default data/cholect50/parquet)")
    p.add_argument("--max-prompt-len", type=int, default=2048, help="verl's filter_overlong_prompts cut-off")
    p.add_argument("--max-new-tokens", type=int, default=256)
    # GRPO (defaults = shared benchmark spec)
    p.add_argument("--rollout-batch-size", type=int, default=16, help="prompts per step")
    p.add_argument("--n-samples", type=int, default=8, help="GRPO group size")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--clip-low", type=float, default=0.2)
    p.add_argument("--clip-high", type=float, default=0.28)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=None, help="cap steps (smoke tests)")
    # Memory / shapes
    p.add_argument("--train-micro-batch", type=int, default=1, help="sequences per forward/backward")
    p.add_argument("--gen-micro-batch", type=int, default=8, help="sequences per rollout forward")
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16",
                   help="base-model dtype. With LoRA the adapter is kept in fp32 regardless; for full-model RL "
                        "bf16 weights can swallow lr=1e-5 updates, so use fp32 + --autocast-bf16 there")
    p.add_argument("--autocast-bf16", action="store_true", help="bf16 autocast for forward passes")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--lora-rank", type=int, default=16,
                   help="16 = LoRA, as the Tinker RL leg; 0 = full-model RL (needs sharding beyond one device)")
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--gen-impl", choices=("static", "hf"), default="static")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (TorchTPU / cuda)")
    # Eval
    p.add_argument("--eval-split", default="val", help="greedy eval before/after on this split ('' to skip)")
    p.add_argument("--eval-every", type=int, default=0)
    # Bookkeeping
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--instance-type", default=os.environ.get("INSTANCE_TYPE"))
    p.add_argument("--hourly-usd", type=float, default=float(os.environ.get("HOURLY_USD", "0") or 0))
    args = p.parse_args(argv)
    total = args.rollout_batch_size * args.n_samples
    for name in ("train_micro_batch", "gen_micro_batch"):
        if total % getattr(args, name):
            p.error(f"--{name.replace('_', '-')} must divide rollout_batch_size*n_samples={total} (static shapes)")
    return args


# ----------------------------------------------------------------------------- model


def load_model(args, backend):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)
    # SDPA / flash kernels are CUDA-centric; eager attention is the portable path.
    if backend.name in ("neuron", "tpu", "tpu-xla", "cpu"):
        model.config._attn_implementation = "eager"
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    ref = None
    if args.lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules="all-linear", task_type="CAUSAL_LM"))
        # Keep the (small) adapter in fp32 even on a bf16 base, so lr=1e-5 updates
        # are not rounded away. peft casts activations to the adapter dtype.
        for prm in model.parameters():
            if prm.requires_grad:
                prm.data = prm.data.float()
        model.print_trainable_parameters()
    elif args.kl_beta > 0:
        ref = copy.deepcopy(model).eval()
        for prm in ref.parameters():
            prm.requires_grad_(False)

    model.to(backend.device)
    if ref is not None:
        ref.to(backend.device)
    if args.compile:
        model = torch.compile(model)
        if ref is not None:
            ref = torch.compile(ref)

    gen_cfg = getattr(model, "generation_config", None)
    tok = getattr(proc, "tokenizer", proc)
    eos = set()
    if gen_cfg is not None and gen_cfg.eos_token_id is not None:
        e = gen_cfg.eos_token_id
        eos |= set(e if isinstance(e, (list, tuple)) else [e])
    if tok.eos_token_id is not None:
        eos.add(tok.eos_token_id)
    return model, ref, proc, sorted(eos)


class _Autocast:
    def __init__(self, backend, enabled):
        self.ctx = None
        if enabled:
            dev_type = "xla" if backend.lazy else backend.device.type
            self.ctx = torch.autocast(device_type=dev_type, dtype=torch.bfloat16)

    def __enter__(self):
        return self.ctx.__enter__() if self.ctx else None

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc) if self.ctx else False


# ----------------------------------------------------------------------------- loop


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def rollout(task, model, examples, args, backend, pad_id, eos_ids, greedy=False):
    """Generate one response per entry of `examples` (already group-expanded)."""
    pass_pos = not task.is_vlm  # VLMs compute their own (3D mRoPE) positions
    outs = []
    model.eval()
    for mb in chunks(examples, args.gen_micro_batch):
        if len(mb) < args.gen_micro_batch:  # keep shapes static: pad with copies, drop after
            mb = mb + [mb[-1]] * (args.gen_micro_batch - len(mb))
        enc = to_device(task.encode(mb), backend.device)
        with _Autocast(backend, args.autocast_bf16):
            if args.gen_impl == "hf":
                resp = generate_hf(model, enc, args.max_new_tokens, args.temperature, pad_id, eos_ids, greedy)
            else:
                resp = generate_static(model, enc, args.max_new_tokens, args.temperature, pad_id, backend,
                                       greedy=greedy, pass_position_ids=pass_pos, eos_ids=eos_ids)
        outs.append(resp.cpu())
    return torch.cat(outs, dim=0)[: len(examples)]


def score_responses(task, examples, resp_ids, eos_ids, args):
    tok = task.tok
    resp, mask, lengths, truncated, first = postprocess(resp_ids, eos_ids, args.max_new_tokens)
    scores = []
    for i, ex in enumerate(examples):
        text = tok.decode(resp[i, : int(first[i])].tolist(), skip_special_tokens=True)
        scores.append(task.score(text, ex))
    return resp, mask, lengths, truncated, scores


def group_advantages(rewards, n):
    adv = []
    for g in chunks(rewards, n):
        mu = sum(g) / len(g)
        sd = statistics.pstdev(g) if len(g) > 1 else 0.0
        adv.extend([(r - mu) / (sd + 1e-6) for r in g])
    return adv


def train_step(task, model, ref, optimizer, examples, resp, mask, adv, args, backend):
    model.train()
    pass_pos = not task.is_vlm
    total_tokens = max(int(mask.sum()), 1)
    stats = {"loss": 0.0, "kl": 0.0}
    for start in range(0, len(examples), args.train_micro_batch):
        sl = slice(start, start + args.train_micro_batch)
        enc = to_device(task.encode(examples[sl]), backend.device)
        r_ids = resp[sl].to(backend.device)
        m = mask[sl].to(backend.device).float()
        a = torch.tensor(adv[sl], device=backend.device, dtype=torch.float32).unsqueeze(1)
        with _Autocast(backend, args.autocast_bf16):
            logp = response_logprobs(model, enc, r_ids, args.temperature, pass_pos)
            if args.kl_beta > 0:
                with torch.no_grad():
                    if ref is not None:
                        ref_logp = response_logprobs(ref, enc, r_ids, args.temperature, pass_pos)
                    else:  # LoRA: reference = base model with the adapter switched off
                        with model.disable_adapter():
                            ref_logp = response_logprobs(model, enc, r_ids, args.temperature, pass_pos)
        old_logp = logp.detach()  # on-policy: one update per rollout batch
        ratio = torch.exp(logp - old_logp)
        pg = -torch.minimum(a * ratio, a * ratio.clamp(1 - args.clip_low, 1 + args.clip_high))
        loss_tok = pg
        if args.kl_beta > 0:
            d = ref_logp - logp
            kl = torch.exp(d) - d - 1  # low_var_kl / k3
            loss_tok = loss_tok + args.kl_beta * kl
            stats["kl"] += float((kl.detach() * m).sum().item()) / total_tokens if not backend.lazy else 0.0
        loss = (loss_tok * m).sum() / total_tokens  # token-mean over the full rollout batch
        loss.backward()
        backend.mark_step()
        if not backend.lazy:  # .item() on XLA forces a sync per micro-batch; skip it there
            stats["loss"] += float(loss.detach().item())
    gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
    backend.optimizer_step(optimizer)
    optimizer.zero_grad(set_to_none=True)
    stats["grad_norm"] = float(gnorm.item())
    return stats


def evaluate(task, model, val, args, backend, pad_id, eos_ids):
    if not val:
        return None
    resp_ids = rollout(task, model, val, args, backend, pad_id, eos_ids, greedy=True)
    _, _, _, _, scores = score_responses(task, val, resp_ids, eos_ids, args)
    return {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    backend = get_backend(args.backend)
    platform_name = args.platform or args.backend
    print(f"[grpo] backend={backend.describe()}", flush=True)
    compat = []
    if backend.device.type == "neuron":
        # Qwen3.5's data-dependent grid_thw / mRoPE / placeholder helpers -> CPU (common/neuron_compat.py).
        from common import neuron_compat

        compat = neuron_compat.install()

    t_load = time.perf_counter()
    model, ref, proc, eos_ids = load_model(args, backend)
    task = make_task(args.task, proc, args.max_prompt_len, args.data)
    pad_id = task.tok.pad_token_id
    train = task.load("train")
    val = task.load(args.eval_split) if args.eval_split else []
    load_s = time.perf_counter() - t_load

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    steps_per_epoch = len(train) // args.rollout_batch_size
    total_steps = steps_per_epoch * args.epochs
    if args.num_steps:
        total_steps = min(total_steps, args.num_steps)
    print(f"[grpo] {len(train)} prompts -> {total_steps} steps "
          f"({args.rollout_batch_size} prompts x {args.n_samples} samples)", flush=True)

    steps_path = os.path.join(args.out_dir, "steps.jsonl")
    steps_f = open(steps_path, "w", encoding="utf-8")
    history, evals = [], []
    t_run = time.perf_counter()

    if val:
        e = evaluate(task, model, val, args, backend, pad_id, eos_ids)
        evals.append({"step": 0, **e})
        print(f"[eval] step 0 {e}", flush=True)

    for step, prompts in enumerate(task.batches(train, args.rollout_batch_size, args.seed, args.epochs), start=1):
        if step > total_steps:
            break
        timing = {}
        examples = [ex for ex in prompts for _ in range(args.n_samples)]  # group-contiguous
        with backend.timed(timing, "rollout_s"):
            resp_ids = rollout(task, model, examples, args, backend, pad_id, eos_ids)
        resp, mask, lengths, truncated, scores = score_responses(task, examples, resp_ids, eos_ids, args)
        rewards = [s["reward"] for s in scores]
        adv = group_advantages(rewards, args.n_samples)
        with backend.timed(timing, "train_s"):
            stats = train_step(task, model, ref, optimizer, examples, resp, mask, adv, args, backend)

        gen_tokens = int(lengths.sum())
        rec = {
            "step": step,
            **{k: sum(s[k] for s in scores) / len(scores) for k in scores[0]},
            "response_len_mean": gen_tokens / len(examples),
            "truncated_frac": float(truncated.float().mean()),
            "zero_std_groups": sum(1 for g in chunks(rewards, args.n_samples) if max(g) == min(g)),
            **stats,
            **timing,
            "step_s": timing["rollout_s"] + timing["train_s"],
            "gen_tok_per_s": gen_tokens / max(timing["rollout_s"], 1e-9),
            "mem_gb": backend.memory_gb(),
        }
        history.append(rec)
        steps_f.write(json.dumps(rec) + "\n")
        steps_f.flush()
        print(f"[step {step}/{total_steps}] reward={rec['reward']:.3f} len={rec['response_len_mean']:.1f} "
              f"trunc={rec['truncated_frac']:.2f} rollout={timing['rollout_s']:.1f}s train={timing['train_s']:.1f}s",
              flush=True)

        if val and args.eval_every and step % args.eval_every == 0 and step != total_steps:
            e = evaluate(task, model, val, args, backend, pad_id, eos_ids)
            evals.append({"step": step, **e})
            print(f"[eval] step {step} {e}", flush=True)

    if val:
        e = evaluate(task, model, val, args, backend, pad_id, eos_ids)
        evals.append({"step": len(history), **e})
        print(f"[eval] final {e}", flush=True)
    steps_f.close()

    wall_s = time.perf_counter() - t_run
    write_results(args, platform_name, backend, task, history, evals, wall_s, load_s, compat)
    if args.lora_rank == 0 and os.environ.get("SAVE_MODEL", "0") == "1":
        model.save_pretrained(os.path.join(args.out_dir, "model"))


def write_results(args, platform_name, backend, task, history, evals, wall_s, load_s, compat=()):
    def curve(key):
        return [round(h[key], 4) for h in history if key in h]

    step_times = [h["step_s"] for h in history]
    steady = step_times[2:] or step_times  # first steps include graph compiles
    exp = {
        "experiment": task.name,
        "method": "grpo",
        "job_id": os.environ.get("JOB_ID"),
        "state": "COMPLETED" if history else "FAILED",
        "base_model": args.model,
        "hf_model_id": args.model if "/" in args.model and not os.path.exists(args.model) else None,
        "lora_rank": args.lora_rank or None,
        "learning_rate": args.lr,
        "kl_beta": args.kl_beta,
        "temperature": args.temperature,
        "n_samples_per_prompt": args.n_samples,
        "rollout_batch_size": args.rollout_batch_size,
        "max_output_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_len,
        "steps": len(history),
        "start_score": curve("reward")[0] if history else None,
        "final_score": curve("reward")[-1] if history else None,
        "score_curve": curve("reward"),
        "wall_time_s": round(wall_s, 1),
        "estimated_cost_usd": round(args.hourly_usd * wall_s / 3600, 2) if args.hourly_usd else None,
    }
    for key in task.curve_keys:
        if key == "reward":
            continue
        c = curve(key)
        if c:
            exp[f"start_{key}"], exp[f"final_{key}"], exp[f"{key}_curve"] = c[0], c[-1], c
    if evals:
        exp["eval_curve"] = evals
        exp["start_val_reward"] = evals[0]["reward"]
        exp["final_val_reward"] = evals[-1]["reward"]
    exp["accelerator"] = {
        **backend.describe(),
        "instance_type": args.instance_type,
        "hourly_usd": args.hourly_usd or None,
        "gen_impl": args.gen_impl,
        "dtype": args.dtype,
        "autocast_bf16": args.autocast_bf16,
        "model_load_s": round(load_s, 1),
        "first_step_s": round(step_times[0], 1) if step_times else None,
        "steady_step_s": round(statistics.median(steady), 1) if steady else None,
        "gen_tok_per_s_median": round(statistics.median([h["gen_tok_per_s"] for h in history]), 1) if history else None,
        "peak_mem_gb": max((h["mem_gb"] or 0) for h in history) or None if history else None,
        "neuron_compat_patched": list(compat),
        "python": sys.version.split()[0],
        "host": _platform.node(),
    }
    results = {
        "platform": platform_name,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiments": {task.name: exp},
        "findings": [],
    }
    path = os.path.join(args.out_dir, "final_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[grpo] wrote {path}", flush=True)


if __name__ == "__main__":
    main()

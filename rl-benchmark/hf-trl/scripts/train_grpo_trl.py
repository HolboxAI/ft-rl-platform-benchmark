#!/usr/bin/env python3
"""CholecT50 GRPO with Hugging Face TRL (`GRPOTrainer`), on whatever device accelerate picks.

On a Trainium host with torch-neuronx >= 2.14 (native TorchNeuron), accelerate
>= 1.15 selects `torch.device("neuron")` by itself (`is_neuron_available()`), so
this is stock TRL with no Neuron-specific code. That is the thing under test.

Same task, data, reward and hyperparameters as the other CholecT50 legs:
  * model    merged SFT checkpoint `cholec-sft-qwen3.5-4b` (Qwen3.5-4B VLM) + LoRA 16
  * data     verl parquet, 256 train / 32 val frames; prompts > 2048 tokens dropped (as verl)
  * GRPO     16 frames x 8 samples = 128 completions per optimizer step, 16 steps (1 epoch),
             temp 0.7, 256 tokens, lr 1e-5, KL 0.04, clip 0.2 / 0.28, group-std advantages
  * reward   0.4 * triplet F1 + 0.6 * mean(instrument, verb, target F1)
  * eval     greedy, 32 val frames, before and after

    accelerate launch --num_processes 1 hf-trl/scripts/train_grpo_trl.py \
        --model ~/models/cholec-sft-qwen3.5-4b --out-dir hf-trl/logs/run1 [--max-steps 1]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from common.rewards.cholect50 import _f1, parse_triplets, reward_single  # noqa: E402

PROMPT = (
    "You are a surgical activity recognition model. List every surgical action "
    "triplet (instrument-verb-target) visible in the endoscopic frame. Reply only "
    "as comma-separated triplets, each formatted instrument-verb-target. If no "
    "action is visible, reply 'none'."
)
CHAT_KW = {"enable_thinking": False}


def completion_text(c) -> str:
    """TRL passes conversational completions as [{"role": "assistant", "content": ...}]."""
    if isinstance(c, str):
        return c
    content = c[0]["content"] if c else ""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def cholect50_reward(completions, gt, **kwargs):
    return [reward_single(parse_triplets(completion_text(c)), json.loads(g)) for c, g in zip(completions, gt)]


def triplet_f1(completions, gt, **kwargs):
    """Logged alongside the reward (weight 0 in the trainer)."""
    out = []
    for c, g in zip(completions, gt):
        pred, g = parse_triplets(completion_text(c)), json.loads(g)
        out.append(_f1(pred, set(g.get("triplets", []))) if pred is not None else 0.0)
    return out


def load_split(data_dir, split, processor, max_prompt_len):
    """verl parquet -> rows {prompt, image, gt}; drops prompts longer than max_prompt_len (verl's filter)."""
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(os.path.join(data_dir, f"{split}.parquet"))
    rows, dropped = [], 0
    for _, r in df.iterrows():
        cell = r["images"][0]
        img = (cell if isinstance(cell, Image.Image) else Image.open(io.BytesIO(cell["bytes"]))).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT_KW)
        n = processor(text=[text], images=[img], return_tensors="pt")["input_ids"].shape[1]
        if n > max_prompt_len:
            dropped += 1
            continue
        rows.append({"prompt": [{"role": "user", "content": PROMPT}], "image": img,
                     "gt": r["reward_model"]["ground_truth"]})
    print(f"[data] {split}: {len(rows)} rows, dropped {dropped} over {max_prompt_len} tokens", flush=True)
    return rows


@torch.no_grad()
def evaluate(model, processor, rows, device, max_new_tokens):
    """Greedy val reward, one frame at a time (same metric as verl / Miles / NeMo-RL)."""
    model.eval()
    tot = {"reward": 0.0, "triplet_f1": 0.0, "parse_ok": 0.0}
    for r in rows:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT_KW)
        enc = processor(text=[text], images=[r["image"]], return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        resp = processor.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred, gt = parse_triplets(resp), json.loads(r["gt"])
        tot["reward"] += reward_single(pred, gt)
        tot["triplet_f1"] += _f1(pred, set(gt.get("triplets", []))) if pred is not None else 0.0
        tot["parse_ok"] += float(pred is not None)
    model.train()
    return {k: round(v / len(rows), 4) for k, v in tot.items()} | {"n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/cholec-sft-qwen3.5-4b"))
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data", "cholect50", "parquet"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-prompt-len", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--steps-per-generation", type=int, default=None,
                    help="micro-batches per generation call; lower it to generate in smaller chunks (less memory). "
                         "Default = all completions of an optimizer step in one generate call per process")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1, help="-1 = one epoch (16 steps)")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-compat", action="store_true", help="do not install neuron_compat")
    ap.add_argument("--hourly-usd", type=float, default=float(os.environ.get("HOURLY_USD", "0") or 0))
    ap.add_argument("--platform", default="hf-trl-trainium")
    args = ap.parse_args()

    from accelerate import PartialState
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    state = PartialState()
    device = state.device
    nproc = state.num_processes
    os.makedirs(args.out_dir, exist_ok=True)
    if state.is_main_process:
        print(f"[device] accelerate picked {device} ({state.distributed_type}, {nproc} process(es))", flush=True)
    compat = []
    if device.type == "neuron" and not args.no_compat:
        # Qwen3.5's data-dependent grid_thw / mRoPE index helpers -> CPU (TorchNeuron cannot lower
        # repeat_interleave with tensor repeats). See neuron_compat.py.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from common import neuron_compat

        compat = neuron_compat.install(verbose=state.is_main_process)

    completions_per_step = args.prompts_per_step * args.num_generations          # 128
    assert completions_per_step % nproc == 0
    grad_accum = completions_per_step // nproc                                     # per_device_train_batch_size = 1
    steps_per_gen = args.steps_per_generation or grad_accum

    processor = AutoProcessor.from_pretrained(args.model)
    t_load = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    load_s = time.perf_counter() - t_load
    train_rows = load_split(args.data_dir, "train", processor, args.max_prompt_len)
    val_rows = [] if args.skip_eval else load_split(args.data_dir, "val", processor, args.max_prompt_len)

    from datasets import Dataset

    train_ds = Dataset.from_list(train_rows)

    evals = []
    t0 = time.perf_counter()
    if val_rows and state.is_main_process:
        model.to(device)
        e = evaluate(model, processor, val_rows, device, args.max_new_tokens)
        evals.append({"step": 0, **e})
        print(f"[eval] before: {e}", flush=True)
    state.wait_for_everyone()

    config = GRPOConfig(
        output_dir=os.path.join(args.out_dir, "trainer"),
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        weight_decay=0.1,
        adam_beta2=0.98,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        steps_per_generation=steps_per_gen,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        beta=args.beta,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="dapo",          # token-level normalisation over active tokens (closest to verl/Miles token-mean)
        scale_rewards="group",     # (r - mean) / std within each group of 8
        reward_weights=[1.0, 0.0],  # triplet_f1 is logged only
        chat_template_kwargs=CHAT_KW,
        gradient_checkpointing=True,
        bf16=True,
        use_vllm=False,            # vLLM on Neuron is a separate image; stock HF generate here
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=1234,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[cholect50_reward, triplet_f1],
        args=config,
        train_dataset=train_ds,
        processing_class=processor,
        peft_config=LoraConfig(r=args.lora_rank, lora_alpha=32, lora_dropout=0.0,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    t_train = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t_train

    if val_rows and state.is_main_process:
        e = evaluate(trainer.model, processor, val_rows, device, args.max_new_tokens)
        evals.append({"step": "final", **e})
        print(f"[eval] after: {e}", flush=True)
    wall = time.perf_counter() - t0

    if not state.is_main_process:
        return
    hist = [h for h in trainer.state.log_history if "reward" in h]
    with open(os.path.join(args.out_dir, "log_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    curve = [round(h["reward"], 4) for h in hist]
    f1_key = next((k for k in (hist[0] if hist else {}) if k.startswith("rewards/triplet_f1")), None)

    def ver(pkg):
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:
            return None

    exp = {
        "experiment": "cholect50_rl",
        "method": "grpo",
        "framework": f"Hugging Face TRL {ver('trl')} GRPOTrainer",
        "state": "COMPLETED" if hist else "UNKNOWN",
        "base_model": args.model,
        "hf_model_id": None,
        "lora_rank": args.lora_rank,
        "learning_rate": args.lr,
        "kl_beta": args.beta,
        "temperature": args.temperature,
        "n_samples_per_prompt": args.num_generations,
        "rollout_batch_size": args.prompts_per_step,
        "max_output_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_len,
        "dataset_size": len(train_rows),
        "val_dataset_size": len(val_rows),
        "steps": trainer.state.global_step,
        "score_curve": curve,
        "start_score": curve[0] if curve else None,
        "final_score": curve[-1] if curve else None,
        "triplet_f1_curve": [round(h[f1_key], 4) for h in hist] if f1_key else [],
        "kl_curve": [round(h.get("kl", 0.0), 5) for h in hist],
        "completion_len_curve": [round(h.get("completions/mean_length", 0.0), 1) for h in hist],
        "eval_curve": evals,
        "start_val_reward": evals[0]["reward"] if evals else None,
        "final_val_reward": evals[-1]["reward"] if len(evals) > 1 else None,
        "wall_time_s": round(wall, 1),
        "train_time_s": round(train_s, 1),
        "estimated_cost_usd": round(args.hourly_usd * wall / 3600, 2) if args.hourly_usd else None,
        "accelerator": {
            "backend": str(device.type),
            "device": str(device),
            "distributed_type": str(state.distributed_type),
            "num_processes": nproc,
            "torch": torch.__version__,
            "torch_neuronx": ver("torch-neuronx"),
            "transformers": ver("transformers"),
            "trl": ver("trl"),
            "peft": ver("peft"),
            "accelerate": ver("accelerate"),
            "instance_type": os.environ.get("INSTANCE_TYPE"),
            "hourly_usd": args.hourly_usd or None,
            "gen_impl": "transformers.generate (TRL default)",
            "steps_per_generation": steps_per_gen,
            "gradient_accumulation_steps": grad_accum,
            "model_load_s": round(load_s, 1),
            "step_times_s": [round(h["step_time"], 1) for h in hist if "step_time" in h] or None,
            "neuron_compat_patched": compat,
        },
    }
    results = {"platform": args.platform,
               "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "experiments": {"cholect50_rl": exp},
               "findings": [f"accelerate selected device {device}; TRL / transformers / peft are stock.",
                            "Only Neuron-specific code: neuron_compat runs Qwen3.5's grid_thw / mRoPE index helpers on CPU "
                            f"({len(compat)} functions) because TorchNeuron cannot lower repeat_interleave with tensor repeats."]}
    with open(os.path.join(args.out_dir, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'final_results.json')}", flush=True)


if __name__ == "__main__":
    main()

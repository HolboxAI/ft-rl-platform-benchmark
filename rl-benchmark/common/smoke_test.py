#!/usr/bin/env python3
"""Phase 0 smoke test: can this accelerator run the pieces CholecT50 GRPO needs?

Uses the real run's model, data and tensor shapes, so a pass here means the
training run's graphs compile. Checks, in order, with wall time per stage
(the first call of each shape includes the compile):

  1. device      a bf16 matmul
  2. load        CholecT50 SFT checkpoint (Qwen3.5-4B VLM) + processor
  3. forward     one val frame, prompt padded to 2048 tokens (twice: compile vs cached)
  4. train       LoRA-16 forward/backward + AdamW step (twice)
  5. generate    16 tokens with the static-shape sampler, decoded and scored with the triplet reward

Writes a JSON summary; exit code != 0 on the first failure, and the failing
stage is named in the summary. That is itself a result.

    python -m common.smoke_test --backend neuron --out trainium/logs/smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import torch

from common.grpo.device import BACKENDS, get_backend
from common.grpo.rollout import generate_static, postprocess, response_logprobs
from common.grpo.tasks import CholecT50Task, to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--model", default=os.path.expanduser("~/models/cholec-sft-qwen3.5-4b"))
    p.add_argument("--data", default=None, help="verl parquet dir (default data/cholect50/parquet)")
    p.add_argument("--max-prompt-len", type=int, default=2048)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    summary = {"backend": args.backend, "model": args.model, "stages": {}, "ok": False}
    stage = "init"
    try:
        backend = get_backend(args.backend)
        summary["describe"] = backend.describe()
        if backend.device.type == "neuron":
            from common import neuron_compat

            summary["neuron_compat_patched"] = neuron_compat.install()

        stage = "device"
        t = time.perf_counter()
        # Random data is drawn on the host: on-device RNG recurses in torch-neuronx 2.14.3 (see device.py).
        a = torch.randn(2048, 2048, dtype=torch.bfloat16).to(backend.device)
        c = (a @ a).float().abs().mean()
        backend.mark_step()
        summary["matmul_check"] = float(c.item())
        summary["stages"]["device_s"] = round(time.perf_counter() - t, 3)

        stage = "load"
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t = time.perf_counter()
        proc = AutoProcessor.from_pretrained(args.model)
        dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)
        model.config._attn_implementation = "eager"
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.lora_rank:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=32, lora_dropout=0.0,
                                                     target_modules="all-linear", task_type="CAUSAL_LM"))
            for prm in model.parameters():
                if prm.requires_grad:
                    prm.data = prm.data.float()
        model.to(backend.device)
        task = CholecT50Task(proc, args.max_prompt_len, args.data)
        ex = task.load("val")[0]
        enc = to_device(task.encode([ex]), backend.device)
        summary["prompt_tokens"] = ex.meta["prompt_tokens"]
        summary["stages"]["load_s"] = round(time.perf_counter() - t, 3)

        stage = "forward"
        model.eval()
        for i in range(2):
            with backend.timed(summary["stages"], f"forward_{i}_s"):
                with torch.no_grad():
                    model(**enc, use_cache=False, logits_to_keep=1)
                backend.mark_step()

        stage = "train"
        model.train()
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-5)
        target = torch.randint(0, task.tok.vocab_size, (1, 16)).to(backend.device)
        for i in range(2):
            with backend.timed(summary["stages"], f"train_step_{i}_s"):
                loss = -response_logprobs(model, enc, target, temperature=0.7, pass_position_ids=False).mean()
                loss.backward()
                backend.mark_step()
                backend.optimizer_step(opt)
                opt.zero_grad(set_to_none=True)
        summary["loss"] = float(loss.detach().item())

        stage = "generate"
        model.eval()
        eos = {task.tok.eos_token_id}
        gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        eos |= set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos] if gen_eos is not None else [])
        with backend.timed(summary["stages"], "generate_16tok_s"):
            resp = generate_static(model, enc, 16, 0.7, task.tok.pad_token_id, backend,
                                   greedy=True, pass_position_ids=False, eos_ids=sorted(eos))
        r, _, _, _, first = postprocess(resp, eos, 16)
        text = task.tok.decode(r[0, : int(first[0])].tolist(), skip_special_tokens=True)
        summary["sample"] = text
        summary["sample_score"] = task.score(text, ex)
        summary["peak_mem_gb"] = backend.memory_gb()
        summary["ok"] = True
        stage = "done"
    except Exception as e:  # the failing stage is the finding
        summary["failed_stage"] = stage
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["traceback"] = traceback.format_exc()[-4000:]
    print(json.dumps({k: v for k, v in summary.items() if k != "traceback"}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()

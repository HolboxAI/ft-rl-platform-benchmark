"""Gate test 2: does the Qwen3.5 Gated DeltaNet LLM (text only) compile on neuron?

The vision tower is known to fail on ``repeat_interleave`` (data-dependent
output shape). This test isolates the text decoder, which is the real #1 risk:
24 Gated DeltaNet linear-attention layers + 8 full-attention layers.

Stages (each prints OK / FAIL):
  1. load model (text decoder only path)
  2. move to `neuron`
  3. eager forward on a text sample (no image)
  4. eager backward on the same sample

Written in Simplified Technical English.
"""
from __future__ import annotations

import os
import sys

import torch

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/ubuntu/model/cholec-sft-qwen3.5-4b")
DEVICE = os.environ.get("DEVICE", "neuron")


def log(*a):
    print(*a, flush=True)


def main():
    import torch_neuronx  # noqa: F401  registers `neuron`
    log("torch", torch.__version__)
    log("stage 0: neuron device ->", torch.device(DEVICE))

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    log("stage 1: model loaded ->", type(model).__name__)

    model.to(DEVICE)
    log("stage 2: model moved to neuron OK")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(
        "grasper-retract-gallbladder, hook-dissect-cystic_duct",
        return_tensors="pt",
    )["input_ids"].to(DEVICE)

    model.eval()
    try:
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
        log("stage 3: text-only forward OK, loss =", float(out.loss))
    except Exception as e:  # noqa: BLE001
        log(f"stage 3 FAIL (forward): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

    model.train()
    try:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        log("stage 4: backward OK")
    except Exception as e:  # noqa: BLE001
        log(f"stage 4 FAIL (backward): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(4)

    log("=== LLM STAGES OK ===")


if __name__ == "__main__":
    main()

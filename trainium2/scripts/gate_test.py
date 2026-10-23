"""Gate test: does Qwen3.5-4B VLM (Gated DeltaNet) run on the `neuron` device?

This is the go/no-go for Experiment A. It loads the SFT checkpoint, moves it to
the `neuron` device, and runs one forward and one backward pass. If this fails,
the full TRL GRPO run cannot happen regardless of TRL, hyperparameters, or data.

Stages (each prints OK / FAIL and stops on the first hard failure):
  1. load processor + model from the checkpoint dir
  2. move model to `neuron` (bf16)
  3. eager forward on one real image sample
  4. eager backward on the same sample
  5. torch.compile(backend="neuron") forward (optional, off by default)

Written in Simplified Technical English.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/ubuntu/model/cholec-sft-qwen3.5-4b")
DATA = os.environ.get("DATA", "/home/ubuntu/bench/cholect50_train.jsonl")
DEVICE = os.environ.get("DEVICE", "neuron")
DO_COMPILE = os.environ.get("DO_COMPILE", "0") == "1"


def log(*a):
    print(*a, flush=True)


def load_sample(data_path: str):
    with open(data_path) as f:
        row = json.loads(f.readline())
    content = row["messages"][0]["content"]
    text = None
    image = None
    for item in content:
        if item["type"] == "text":
            text = item["text"]
        elif item["type"] == "image_url":
            raw = item["image_url"]
            # Accept a raw data: URI or a bare base64 string.
            if isinstance(raw, str) and raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            image = base64.b64decode(raw)
    return text, image


def main():
    import torch
    import torch_neuronx  # noqa: F401  registers `neuron`
    log("torch", torch.__version__)
    log("stage 0: neuron device ->", torch.device(DEVICE))

    # --- stage 1: load processor + model ---
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=False)
    log("stage 1: processor loaded:", type(proc).__name__)

    # qwen3_5 is a vision-language model; transformers 5.x maps it to the
    # image-text-to-text auto class (AutoModelForConditionalGeneration is gone).
    from transformers import AutoModelForImageTextToText
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        log("stage 1: model loaded ->", type(model).__name__)
    except Exception as e:  # noqa: BLE001
        log(f"stage 1 FAIL (model load): {type(e).__name__}: {e}")
        sys.exit(1)

    # --- stage 2: move to neuron ---
    try:
        model.to(DEVICE)
        log("stage 2: model moved to", DEVICE, "OK")
    except Exception as e:  # noqa: BLE001
        log(f"stage 2 FAIL: {type(e).__name__}: {e}")
        sys.exit(2)

    # --- build one real sample ---
    text, image_bytes = load_sample(DATA)
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]},
    ]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[prompt], images=[image], return_tensors="pt")
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(DEVICE)

    # --- stage 3: eager forward ---
    model.eval()
    try:
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        log("stage 3: eager forward OK, loss =", float(out.loss))
    except Exception as e:  # noqa: BLE001
        log(f"stage 3 FAIL (forward): {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(3)

    # --- stage 4: eager backward ---
    model.train()
    try:
        out = model(**inputs, labels=inputs["input_ids"])
        out.loss.backward()
        log("stage 4: eager backward OK")
    except Exception as e:  # noqa: BLE001
        log(f"stage 4 FAIL (backward): {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(4)

    # --- stage 5: torch.compile (optional) ---
    if DO_COMPILE:
        try:
            import torch._dynamo
            torch._dynamo.config.automatic_dynamic_shapes = False
            model.eval()
            cm = torch.compile(model, backend="neuron")
            with torch.no_grad():
                out = cm(**inputs, labels=inputs["input_ids"])
            log("stage 5: torch.compile forward OK, loss =", float(out.loss))
        except Exception as e:  # noqa: BLE001
            log(f"stage 5 FAIL (compile): {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            sys.exit(5)

    log("=== ALL STAGES OK ===")


if __name__ == "__main__":
    main()

"""TRL GRPO (RLVR) on CholecT50 for Trainium2 (Experiment A).

Mirrors the Miles/SkyRL run on the same task, model, data, reward, and
hyperparameters. The only changes from the CUDA runs are the Trainium-specific
TRL knobs: static cache, bf16 model, transformers generation (vLLM does not run
on Neuron), and torch.compile with the `neuron` backend.

Run (single process on one NeuronCore first):
    NEURON_RT_VISIBLE_CORES=0 python grpo_train.py
"""
from __future__ import annotations

import base64
import io
import json
import os

import torch
from datasets import Dataset
from PIL import Image
from trl import GRPOConfig, GRPOTrainer

from peft import LoraConfig

from reward_tr import reward_func

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/ubuntu/model/cholec-sft-qwen3.5-4b")
TRAIN = os.environ.get("TRAIN", "/home/ubuntu/bench/cholect50_train.jsonl")
VAL = os.environ.get("VAL", "/home/ubuntu/bench/cholect50_val.jsonl")
DEVICE = os.environ.get("DEVICE", "neuron")
SMOKE = os.environ.get("SMOKE", "0") == "1"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "0") == "1"
# KL coefficient. Reference (verl) is 0.04, which makes TRL load a SECOND full
# model copy as the reference model (~10 GB on this checkpoint). That does not fit
# one NeuronCore's 25.8 GB HBM, so the smoke run uses beta=0.0 (no ref model, no KL).
BETA = float(os.environ.get("BETA", "0.0"))
USE_LORA = os.environ.get("USE_LORA", "1") == "1"
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))
# Batch / generation geometry, directly overridable. Defaults mirror the verl
# reference (batch 16, n=8) unless SMOKE is set.
PER_DEVICE_BATCH = int(os.environ.get("BATCH", "1" if SMOKE else "16"))
NUM_GENERATIONS = int(os.environ.get("NUM_GEN", "2" if SMOKE else "8"))
GEN_BATCH = int(os.environ.get("GEN_BATCH", "2" if SMOKE else "8"))

# Turn off Dynamo automatic dynamic shapes: the Neuron backend cannot compile
# the resulting graphs.
torch._dynamo.config.automatic_dynamic_shapes = False


def _decode_image(item) -> Image.Image:
    raw = item["image_url"]
    if isinstance(raw, str) and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def load_dataset(path: str) -> Dataset:
    prompts, images, labels = [], [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            content = row["messages"][0]["content"]
            text = None
            img = None
            for item in content:
                if item["type"] == "text":
                    text = item["text"]
                elif item["type"] == "image_url":
                    img = _decode_image(item)
            prompts.append([{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ]}])
            images.append(img)
            labels.append(row["label"])
    return Dataset.from_dict({"prompt": prompts, "image": images, "label": labels})


def _patch_vision_interp():
    """Force Qwen3.5 vision position-interpolation precompute onto CPU.

    ``get_vision_interpolation_indices_and_weights`` uses ``torch.repeat_interleave``
    with a dynamic ``repeats`` tensor. neuronx-cc cannot compile that op (Torch-MLIR
    lowering fails or BIRCodeGenLoop crashes), so the small ``(total_thw, n_taps)``
    index/weight tensors are computed on CPU and moved back. This is a sync, not a
    correctness change: eager mode already falls back to CPU for the same op.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    _orig = mq.get_vision_interpolation_indices_and_weights

    def _patched(grid_thw, *args, **kwargs):
        dev = grid_thw.device
        print(
            "INTERP_PATCH dev=", dev,
            "compiling=", torch._dynamo.is_compiling(),
            flush=True,
        )
        if dev.type == "neuron":
            indices, weights = _orig(grid_thw.to("cpu"), *args, **kwargs)
            return indices.to(dev), weights.to(dev)
        return _orig(grid_thw, *args, **kwargs)

    mq.get_vision_interpolation_indices_and_weights = _patched


# LoRA targets cover every linear projection in the language-model body:
# full-attention layers (q/k/v/o), Gated DeltaNet linear-attention layers
# (in_proj_qkv/z/b/a + out_proj), and the MLPs (gate/up/down). The lm_head,
# embeddings, vision tower and the GDN conv1d are left frozen.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
]


def main():
    import torch_neuronx  # noqa: F401  registers `neuron`

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    # Move to the Neuron device up front so we see immediately whether the
    # weights land on `neuron`; the trainer may re-place them via accelerate.
    model.to(DEVICE)
    print("MODEL_DEVICE", next(model.parameters()).device, flush=True)
    _patch_vision_interp()

    train_dataset = load_dataset(TRAIN)
    eval_dataset = load_dataset(VAL)
    if SMOKE:
        train_dataset = train_dataset.select(range(min(32, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(8, len(eval_dataset))))

    args = GRPOConfig(
        output_dir="/home/ubuntu/grpo_out",
        cache_implementation="static",
        model_init_kwargs={"dtype": "bfloat16"},
        # --- task geometry (verl/Miles match) ---
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=1,
        num_generations=NUM_GENERATIONS,
        num_generations_eval=1,  # greedy-ish eval: 1 completion/prompt (matches verl n=1 eval)
        generation_batch_size=GEN_BATCH,
        max_completion_length=256,
        temperature=0.7,
        top_p=1.0,
        # --- optimizer / algorithm (verl/Miles match) ---
        learning_rate=1e-5,
        beta=BETA,               # KL coefficient (verl kl_loss_coef 0.04)
        num_train_epochs=1,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else None,
        # --- Trainium ---
        torch_compile=TORCH_COMPILE,
        torch_compile_backend="neuron" if TORCH_COMPILE else None,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        # The Qwen3.5 chat template bakes an empty <think> block. Keep it: the
        # reward strips it. Do not let TRL try to detect/rewrite the template.
    )

    lora_config = None
    if USE_LORA:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )

    print("BETA", BETA, "USE_LORA", USE_LORA, "LORA_R", LORA_R, flush=True)

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_func,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("TRAINABLE_PARAMS", trainable, "TOTAL_PARAMS", total,
          "FRAC", round(trainable / total, 5), flush=True)

    print("ACCEL_DEVICE", trainer.accelerator.device, flush=True)
    print("ACCEL_STATE_NEURON", getattr(trainer.accelerator.state, "use_neuron", None), flush=True)

    trainer.train()

    # Release training-time device buffers before eval. The multi-step run's
    # logprob/generation allocations stay cached in the Neuron allocator and
    # starve the eval's fused-LM-head logits (~288 MB) of HBM.
    import gc
    gc.collect()
    torch_neuronx.synchronize()
    torch_neuronx.empty_cache()
    print("POST_TRAIN_MEM_ALLOCATED", torch_neuronx.memory_allocated(), flush=True)

    # Final eval: greedy mean@1 on the val split, same metric as Miles/SkyRL.
    out = trainer.evaluate()
    print("EVAL_RESULT", json.dumps(out))


if __name__ == "__main__":
    main()

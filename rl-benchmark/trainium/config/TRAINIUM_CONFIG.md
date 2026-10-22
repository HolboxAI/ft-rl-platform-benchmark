# Trainium: configuration and findings (CholecT50)

_Fill in the ⏳ cells as runs happen. Strike through superseded values rather
than deleting them, so the history of what was tried stays visible._

## Hardware

| | Value |
|---|---|
| Instance | `trn1.2xlarge` (fallback: `trn2`, 96 GB HBM per chip) |
| Accelerator | 1× Trainium1 chip = 2 NeuronCore-v2, 32 GB HBM (16 GB per core) |
| Cores used | 1 (`NEURON_RT_NUM_CORES=1`) |
| On-demand $/hr | ~1.34 (us-east-1, verify) |
| Account quota | 32 vCPU on-demand Trn in us-east-1 / us-west-2 (checked 2026-10-22): up to 4× trn1.2xlarge |
| Region / AZ | ⏳ |

Memory budget on one NeuronCore (16 GB): Qwen3.5-4B in bf16 is ~8.5 GB of
weights. The LoRA-16 adapter and its fp32 AdamW state add well under 1 GB. The
reference model is the same weights with the adapter disabled, so it costs
nothing extra. Activations at 2,048 + 256 tokens with gradient checkpointing
take the rest. If it runs out of memory:
1. Lower `--gen-micro-batch` to 4. That shrinks the sampler buffer.
2. Move to `trn2`.

Record either change as a deviation. Raising `NEURON_RT_NUM_CORES` does not
help: this trainer does not shard.

## Software

| | Value |
|---|---|
| AMI | Neuron DLAMI, Ubuntu 22.04 — ⏳ AMI id |
| Neuron SDK | ⏳ (from `logs/environment.txt`) |
| torch / torch-xla / torch-neuronx | ⏳ |
| transformers / peft | ⏳ (Qwen3.5 needs transformers ≥ 5.x) |
| Compiler flags | `NEURON_CC_FLAGS=--model-type=transformer --retry_failed_compilation` |
| Compile cache | `/var/tmp/neuron-compile-cache` (cold on first run) |

## CholecT50 GRPO settings

| Parameter | Value | Same as |
|---|---|---|
| Model | merged SFT checkpoint `cholec-sft-qwen3.5-4b` (Qwen3.5-4B VLM) | verl / Miles / SkyRL / NeMo-RL |
| Method | LoRA rank 16, all linear layers; reference = adapter disabled | Tinker RL leg |
| Data | 256 train / 32 val frames (verl parquet); prompts > 2048 tokens dropped | verl |
| Frames × samples per step | 16 × 8 = 128 rollouts | all legs |
| Steps | 16 (1 epoch) | Tinker / Miles |
| Temperature / top_p / top_k | 0.7 / 1.0 / off | verl / Miles |
| Max response tokens | 256 | all legs |
| LR / optimizer | 1e-5, AdamW (0.9, 0.98), wd 0.1, constant | Miles |
| KL | 0.04, `low_var_kl` | verl / Miles |
| Clip | 0.2 / 0.28 | Miles |
| Precision | bf16 base weights, fp32 LoRA adapter + optimizer | — |
| Rollout engine | in-process static-shape sampler (no KV cache) | ❌ others use vLLM / SGLang |
| Reward | `common/rewards/cholect50.py` (0.4 × triplet F1 + 0.6 × component F1) | all legs |
| Eval | greedy, 32 val frames, before and after | Miles (val n=32) |

Known risks:
- **Gated-DeltaNet fallback.** Qwen3.5's Gated-DeltaNet layers use CUDA/Triton
  kernels on GPU (`fla`, `causal_conv1d`). On Neuron they must use the
  pure-PyTorch fallback, which may be slow or fail to compile.
- **Vision tower recompute.** The static sampler re-runs the vision tower for
  every generated token: 256 times per rollout.

## Findings log

| Date | Phase | Finding |
|---|---|---|
| ⏳ | | |

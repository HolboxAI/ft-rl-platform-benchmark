# Trainium2 — CholecT50 GRPO / RLVR configuration

Exact configuration for the RL / GRPO stage on Trainium2 (TorchNeuron). The SFT
stage was produced on Tinker; this stage loads that merged checkpoint and runs
TRL GRPO against the triplet-F1 reward.

## Shared

| Parameter | Value |
|---|---|
| Base model | `cholec-sft-qwen3.5-4b` (Qwen3.5-4B VLM, `model_type` `qwen3_5`) |
| SFT checkpoint | Tinker output, LoRA merged (same checkpoint as verl, Miles, SkyRL) |
| Data | CholecT50 — 256 train / 32 val frames (base64-image JSONL) |
| Task | list `instrument-verb-target` action triplets per frame |
| Reward | `0.4 * triplet_F1 + 0.6 * mean(instrument_F1, verb_F1, target_F1)` |
| Precision | bfloat16 (model), eager mode (no torch.compile) |
| Device | `neuron` — single NeuronCore (`NEURON_RT_VISIBLE_CORES=0`) |

## Stack (verified working pins)

| Package | Version | Note |
|---|---|---|
| torch | 2.14.0+cu130 | from the DLAMI venv |
| torch-neuronx | **2.14.3.0.282** | pinned `--no-deps`; AMI-bundled 2.14.3.0.1 crashes on `repeat_interleave` |
| torchvision | 0.29.1 | from the DLAMI venv |
| transformers | 5.18.0 | required for the `qwen3_5` architecture |
| trl | 1.16.0 | GRPO trainer |
| accelerate | 1.15.0 | |
| peft | >=0.15 | LoRA |
| datasets / pillow | >=3 / >=11 | |
| python | 3.14 | |

## RL / GRPO stage (TRL)

### Algorithm

| Parameter | Value |
|---|---|
| method | `GRPOTrainer` |
| `learning_rate` | `1e-5` |
| `beta` (KL) | `0.0` (see note) |
| `num_train_epochs` | `1` |
| `max_steps` | `4` (smoke) / `64` (full 1-epoch = 256 rows ÷ batch 4) |
| `gradient_checkpointing` | `True` |
| `torch_compile` | `False` (eager) |
| `cache_implementation` | `static` |

### LoRA (PEFT)

| Parameter | Value |
|---|---|
| `r` | `16` |
| `lora_alpha` | `32` |
| `lora_dropout` | `0.0` |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj` |
| trainable / total | 32.5M / 5.21B (0.62%) |

### Data / geometry

| Parameter | Value |
|---|---|
| `per_device_train_batch_size` | `1` (smoke) / `4` (full 1-epoch) |
| `per_device_eval_batch_size` | `1` |
| `num_generations` | `8` (smoke) |
| `num_generations_eval` | `1` (greedy mean@1, matches verl n=1 eval) |
| `generation_batch_size` | `8` (must be divisible by `num_generations`) |
| `max_completion_length` | `256` |
| `pad_to_multiple_of` | `256` (full run; keeps the scoring-forward shape constant — issues #22/#23) |
| image size | `854×480` normalized (full run; frames arrive in 854×480 and 1920×1080) |
| smoke train/val rows | 32 / 8 |

### Rollout

| Parameter | Value |
|---|---|
| backend | transformers `model.generate` (vLLM does not run on Neuron) |
| `temperature` | `0.7` |
| `top_p` | `1.0` |

### Hardware / runtime

| Parameter | Value |
|---|---|
| Instance | AWS EC2 `trn2.48xlarge` spot, ap-south-2b |
| Cores | 1 of 48 (`NEURON_RT_VISIBLE_CORES=0`) |
| HBM per core | 25.8 GB |
| Env | `NKI_ENABLE_TRACE_CACHE=0` |
| Smoke train time | ~141 s (4 steps) |
| Full-epoch train time | ~90 min (64 steps) |
| Full-epoch eval time | ~511 s (32 frames) |

## Notes on the two hard memory constraints

1. **KL reference model.** TRL GRPO with `beta != 0` loads a second full copy of
   the model as the reference model (~10 GB). On one NeuronCore this is too much,
   so the smoke uses `beta = 0.0`.
2. **Optimizer states.** Full fine-tune AdamW keeps fp32 `m`/`v` states
   (~2 × 4.95B × 4 B ≈ 40 GB), which cannot fit one core even without the ref
   model. LoRA shrinks the trainable set to 32.5M, so the optimizer states are
   tiny.

A full comparable run needs either multi-core tensor parallelism (NeuronX
Distributed) or a CPU-offload optimizer. Both are open; see
`TRAINIUM_ISSUES_LOG.md`.

## Eval

| Parameter | Value |
|---|---|
| Val split | 8 frames (smoke) |
| Decoding | `num_generations_eval=1` (one completion per prompt) |
| Metric | blended reward (same formula as training) |
| Result | 0.2958 smoke / **0.3573 full 1-epoch** (SFT baseline 0.31875; see `metrics/final_results.json`) |

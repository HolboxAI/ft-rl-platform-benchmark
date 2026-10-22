# TRL on Trainium: CholecT50 settings and findings

## Versions

| | Planned | Observed |
|---|---|---|
| AMI | Neuron DLAMI `torch-neuronx-2.14.0`, Ubuntu 24.04 | ⏳ |
| torch / torch-neuronx | 2.14.0 / 2.14.3 (from the Miles probe on the same AMI) | ⏳ |
| trl / accelerate / transformers | 1.16.0 / 1.15.0 / 5.19.0 | ⏳ |
| peft | latest compatible | ⏳ |
| Instance | trn1.2xlarge (1 chip, 2 NeuronCores, 32 GB), 1 process | ⏳ |

## GRPOConfig (as passed by `train_grpo_trl.py`)

| Field | Value | Why |
|---|---|---|
| `per_device_train_batch_size` | 1 | memory on one NeuronCore |
| `gradient_accumulation_steps` | 128 / num_processes | one optimizer step per 128 completions (16 frames × 8), like the other legs |
| `steps_per_generation` | 16 (smoke / full) | generate 16 completions per `generate` call to bound memory; still on-policy within a step |
| `num_generations` | 8 | GRPO group size |
| `max_completion_length` | 256 | |
| `temperature` / `top_p` / `top_k` | 0.7 / 1.0 / 0 | verl / Miles |
| `beta` | 0.04 | verl / Miles (reference = LoRA adapter disabled) |
| `epsilon` / `epsilon_high` | 0.2 / 0.28 | Miles |
| `loss_type` | `dapo` | token-level, closest to verl/Miles token-mean |
| `scale_rewards` | `group` | (r − mean) / std per group |
| `learning_rate`, scheduler | 1e-5, constant | |
| `optim`, betas, wd | `adamw_torch`, (0.9, 0.98), 0.1 | Miles |
| LoRA | r 16, α 32, all linear, dropout 0 | Tinker leg |
| precision | bf16 weights, `bf16=True` | |
| `gradient_checkpointing` | True | |
| `use_vllm` | False (stock `transformers.generate`) | vLLM-on-Neuron is a separate DLAMI |
| `chat_template_kwargs` | `{"enable_thinking": false}` | |

## Findings log

| Date | Stage | Finding |
|---|---|---|
| 2026-10-22 | (Miles probe, same AMI) | The torch-neuronx 2.14 venv has no `torch_xla`: torch-neuronx is the native TorchNeuron backend there. |
| 2026-10-22 | design | optimum-neuron 0.4.3 has no GRPO trainer and pins XLA-based torch-neuronx 2.8 + transformers 4.57, so it is not used. |
| 2026-10-22 | check (run `trl-trn-20261022-1141`, trn1.2xlarge) | **Native TorchNeuron works on Trn1**: `torch.neuron.is_available()`, 2 devices, tensor op OK. **accelerate 1.15 picked `neuron` by itself.** trl 1.16.0 / transformers 5.19.0 / peft 0.21.2 / accelerate 1.15.0 import. Val split: 28 frames after the 2048-token filter (same as verl). |
| 2026-10-22 | check, model stage | First `generate` failed in the TorchNeuron compiler: `transformers/vision_utils.py:get_vision_interpolation_indices_and_weights` → `torch.repeat_interleave(grid_thw[:, 1], counts)` → `TorchMlirCompilerError: 'stablehlo.concatenate' … tensor<2xi64> vs tensor<1xi64>`. Data-dependent output shape not lowered. Workaround: `scripts/neuron_compat.py` runs the `grid_thw` / mRoPE index helpers on CPU (transformers documents them as untraceable, meant to be precomputed). Rerun `trl-trn-20261022-1141-r2`. |
| 2026-10-22 | check, model stage (`-r2`) | With the CPU index shim the vision tower **compiles (~12 min cold) and runs** on Trainium (405 image features). Next failure: `get_placeholder_mask` counted **203** image tokens on neuron for a frame with **405** (`(input_ids == image_token_id).sum()`). |
| 2026-10-22 | ops stage (`-r3`): 16 ops, neuron vs CPU | **14 / 16 match**: int64 / int32 `==` + `.sum()`, bool `.sum()` / `.long().sum()` / `.int().sum()`, `sum(dim)`, bool cumsum, `argmax`, `nonzero`, `masked_select`, `any`/`all`, `gather`, masked mean, float sum. **Wrong:** `int64.sum()` of large values (330901472 vs 330901475: float32 accumulation, exact only < 2^24) and `int64.cumsum()` of large values (586250 vs 330407798; increments of exactly 255). The standalone image-token count is right, so the in-model 203 comes from another cause. |
| 2026-10-22 | `-r4` | Added op cases for the two suspects (small-value cumsum like position_ids; kernel-cache reuse across `== const` with two constants, which the NKI trace-cache warning hints at). `get_placeholder_mask` → CPU in `neuron_compat`. Running check + 1-step smoke. |
| ⏳ | | |

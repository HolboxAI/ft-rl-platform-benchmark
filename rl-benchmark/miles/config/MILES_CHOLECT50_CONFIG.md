# Miles CholecT50 — reference settings and findings

## Reference run (GPU) — what an accelerator port must match

From `reference_gpu/cholect50_rl_final_results.json` and `run_cholect50_rl.sh`:

| Setting | Value |
|---|---|
| Model | `cholec-sft-qwen3.5-4b` (Tinker SFT, LoRA merged; same checkpoint as verl / SkyRL) |
| Method | GRPO, full model (no LoRA) |
| Train backend / rollout | FSDP + CPU offload / SGLang |
| Data | 256 train / 32 val (verl parquet → Miles JSONL; overlong-prompt filter **not** applied) |
| Prompts × samples per step | 16 × 8 (global batch 128) |
| Steps | 16 (1 epoch) |
| LR / KL | 1e-5 / 0.04 `low_var_kl` |
| Temperature / top_p | 0.7 / 1.0 |
| Max response / prompt tokens | 256 / 2048 |
| Reward | `reward_cholect50.py`: 0.4 × triplet F1 + 0.6 × mean(instrument, verb, target F1) |
| Hardware | p4d.24xlarge spot, 4× A100 used |
| Result | val 0.319 → **0.651** (verl: 0.000 → 0.715); rollout reward 0.337 → 0.537 (peak 0.696) |
| Wall / cost | 4193 s / $11.6 |

Known divergences from verl (from the GPU run): the reward EOS-token bug fixed
mid-study (`run_cholect50_broken.log`), the missing overlong-prompt filter, and a
different SFT baseline (0.319 vs 0.000).

## Probe results

| Date | Chip / instance | Stage reached | First failure | Notes |
|---|---|---|---|---|
| 2026-10-22 | laptop CPU, Python 3.9 (script dry-run only) | `miles_import` | `accelerator` (no CUDA) | Validates the probe only. Miles needs Python ≥ 3.10. |
| ⏳ | TPU (v6e-1 / v7x?) | | | |
| 2026-10-22 | trn1.2xlarge, DLAMI torch-neuronx 2.14.0 (torch 2.14.0, torch-neuronx 2.14.3), run `miles-trn-20261022-1122-r2` | `miles_import`, `train_backends` (torchtitan, megatron import), `import_surface` 480 / 645 modules | `accelerator` (probe expected `torch_xla`; the venv has none: native TorchNeuron), `platform` (`MILES_ACCELERATOR='neuron'; expected cuda or tpu`), `rollout` (no sglang) | SGLang cannot even resolve its deps here (`apache-tvm-ffi==0.1.0b15` not found). Import failures: sglang 51, sglang_router 23, megatron 24, platform ValueError 27. Cost < $0.50. |

## Findings log

| Date | Finding |
|---|---|
| 2026-10-22 | Miles ships an experimental TPU platform (`MILES_ACCELERATOR=tpu`, `miles/platforms/tpu.py`, "TPU v7x through torch_tpu", static shapes only). Training side only; rollout remains SGLang. Not in the docs. |
| 2026-10-22 | No Trainium/Neuron platform; `_select_platform` raises for anything but `cuda`/`tpu`. |
| 2026-10-22 | Trainium probe: Miles imports next to torch-neuronx, but the platform hook rejects `neuron` and SGLang is uninstallable on the Neuron venv. The DLAMI does ship `torchtitan-neuron` 0.2.2 (Miles has a torchtitan backend), and AWS publishes vLLM-on-Neuron DLAMIs: the two pieces a future port would build on. |
| 2026-10-22 | Static audit: 552 NVIDIA-specific call sites (CUDA 141, NCCL 26, kernels 62, NVIDIA stack 323) + 67 SGLang call sites in production code; 0 JAX. |

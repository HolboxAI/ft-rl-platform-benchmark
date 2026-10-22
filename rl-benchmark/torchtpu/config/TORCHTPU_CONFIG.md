# TorchTPU: configuration and findings (CholecT50)

_Fill in the ⏳ cells as runs happen._

## Hardware

| | Value |
|---|---|
| TPU slice | `v6e-1` (1 Trillium chip, 32 GB HBM); fallback `v6e-4` |
| Runtime image | ⏳ (`v2-alpha-tpuv6e`?) |
| Project / zone | ⏳ |
| On-demand $/hr | ⏳ (check Cloud TPU pricing for the zone) |
| Spot? | ⏳ |

Qwen3.5-4B in bf16 (~8.5 GB) + LoRA 16 + activations at 2,048 + 256 tokens fits
a 32 GB v6e chip with room to spare. A 16 GB v5e chip is tight; if you must use
one, lower `--gen-micro-batch` to 4 and record it.

## Software

| | TorchTPU path | PyTorch/XLA path |
|---|---|---|
| torch | ⏳ | ⏳ |
| torch_tpu / torch_xla | ⏳ | ⏳ |
| libtpu | ⏳ | ⏳ |
| transformers / peft | ⏳ | ⏳ |
| Install command used | ⏳ | `pip install torch 'torch_xla[tpu]'` |
| Graph mode | eager; `torch.compile` with `COMPILE=1` | lazy tensors + `mark_step` |

## CholecT50 GRPO settings

Identical to the Trainium leg (see
[`../../trainium/config/TRAINIUM_CONFIG.md`](../../trainium/config/TRAINIUM_CONFIG.md#cholect50-grpo-settings)):
- Model: merged SFT checkpoint, LoRA 16.
- Data: 256 / 32 frames, 16 frames × 8 samples per step, 16 steps.
- Sampling: temperature 0.7, 256 response tokens.
- Optimisation: lr 1e-5, KL 0.04 `low_var_kl`, clip 0.2 / 0.28.
- Precision: bf16 base, fp32 adapter.
- Rollouts: static-shape sampler. Eval: greedy on 32 val frames, before and after.

Run the three paths back-to-back on the same VM, with nothing else on it:

| Run | `--backend` | Extra |
|---|---|---|
| TorchTPU eager | `tpu` | — |
| TorchTPU compile | `tpu` | `COMPILE=1` (`--compile`) |
| PyTorch/XLA | `tpu-xla` | `PJRT_DEVICE=TPU` |

## Findings log

| Date | Path | Finding |
|---|---|---|
| ⏳ | | |

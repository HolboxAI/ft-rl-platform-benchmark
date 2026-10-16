# verl (ByteDance) — GRPO / RLVR on the Tinker SFT checkpoint

This folder documents the CholecT50 surgical action-triplet pipeline run on
**verl** (ByteDance's RL framework) — a GRPO / RLVR stage that loads the
**Tinker-produced SFT checkpoint** and trains against the same custom
triplet-F1 reward. It is the fourth platform in this benchmark's CholecT50 leg
(after Baseten 🚫, Together AI ❌, Tinker ✅), and the first that runs **full-model
RL (no LoRA)** on raw CUDA hardware rather than a managed fine-tuning API.

## The task

| | |
|---|---|
| Dataset | CholecT50 — endoscopic surgery frames |
| Task | list every `instrument-verb-target` action triplet visible in a frame |
| Model | `Qwen/Qwen3.5-4B` (vision-language) |
| SFT checkpoint | produced on Tinker, merged + thinking-disabled, on S3 |
| RL stage | GRPO (RLVR) on verl, full-model FSDP |

## Verdict

| Requirement | verl |
|---|---|
| Full-model RL with a custom reward (RLVR / GRPO) | ✅ |
| Load an external (Tinker) SFT checkpoint | ✅ |
| before/after F1 demo | ✅ 0.0 → 0.715 blended reward |

## Before / After (val split, n=28 frames, greedy)

| Metric | Start (thinking-disabled SFT) | After RL / GRPO |
|---|---|---|
| blended reward | 0.000 | **0.715** |

`blended reward = 0.4 * triplet_F1 + 0.6 * mean(instrument, verb, target F1)`

> **Note on the "before".** The Tinker SFT was trained in thinking mode. Once
> thinking is disabled (a required fix for the RLVR triplet format — see the
> Tinker experience report), that checkpoint emits **empty output** and scores
> **0.0**. GRPO then teaches the model to emit the triplet format directly,
> reaching **0.715** on the val split after a single epoch. This is therefore a
> "0 → X" demo, but the 0.0 baseline is a thinking-disabled SFT, **not** a
> thinking-enabled SFT-at-0.364 like the Tinker leg — the numbers are not directly
> comparable to Tinker's SFT→RL refinement.

Training reward (`critic/rewards/mean`) over the 14 GRPO steps:

```
0.001 0.268 0.429 0.312 0.350 0.389 0.453 0.550 0.478 0.436 0.441 0.502 0.601 0.548
```

## Contents

| Path | What it holds |
|---|---|
| `config/VERL_CONFIG.md` | every hyperparameter for the RL stage |
| `scripts/reward.py` | triplet-F1 reward, verl `compute_score` entrypoint |
| `scripts/cholec_data.py` | CholecT50 → verl parquet (train/val) |
| `scripts/train.sh` | full verl launch command (all Hydra overrides) |
| `scripts/Dockerfile` | image build (applies the three Qwen3.5 patches) |
| `scripts/patch_qwen35.py`, `patch_tensordict.py`, `patch_fix.py` | verl patches for Qwen3.5 3D mRoPE position_ids |
| `metrics/` | `final_results.json` + `run_manifest.json` |

## Key facts

- **Full-model RL, not LoRA.** verl runs FSDP over all parameters (bfloat16),
  with parameter + optimizer offload to CPU so a 4B VLM fits on 4× A10G (24 GB).
- **Qwen3.5 crashes vanilla verl.** The model's 3D mRoPE `position_ids` trip three
  bugs in the base `verlai/verl:uv.cu130` image; the three patches in `scripts/`
  are applied by the `Dockerfile`.
- **Offline by design.** The image ships a 16 GB uv cache; training runs with
  `--frozen --offline` and never touches the HF hub.
- **Checkpoint kind.** verl writes full HF weights + optimizer state
  (`global_step_14/`), unlike Tinker's `weights/` vs `sampler_weights/` split.
- **Data is a subset.** This run used 256 train / 32 val frames (Tinker's RL used
  256 train but evaluated on 300 val); after verl's `filter_overlong_prompts`
  the effective sizes are 235 train / 28 val.

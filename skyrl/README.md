# SkyRL (NovaSky) — GRPO / RLVR on the Tinker SFT checkpoint

This folder documents the CholecT50 surgical action-triplet pipeline run on
**SkyRL** (NovaSky-AI's modular RL framework built on verl) — a GRPO / RLVR
stage that loads the **Tinker-produced SFT checkpoint** and trains against the
same custom triplet-F1 reward. It is the fifth platform in this benchmark's
CholecT50 leg (after Baseten 🚫, Together AI ❌, Tinker ✅, verl ✅), and it
mirrors the verl leg: **full-model RL (no LoRA)** on FSDP2 + vLLM, self-managed
on raw EC2 rather than a managed fine-tuning API.

## The task

| | |
|---|---|
| Dataset | CholecT50 — endoscopic surgery frames |
| Task | list every `instrument-verb-target` action triplet visible in a frame |
| Model | `Qwen/Qwen3.5-4B` (vision-language) |
| SFT checkpoint | produced on Tinker, merged (`~/cholec/merged`) |
| RL stage | GRPO (RLVR) on SkyRL, full-model FSDP2 |

## Verdict

| Requirement | SkyRL |
|---|---|
| Full-model RL with a custom reward (RLVR / GRPO) | ✅ |
| Load an external (Tinker) SFT checkpoint | ✅ |
| before/after F1 demo | ❌ — reward collapsed to 0.0 (see below) |

## Result — collapse (not a reproduction)

SkyRL **did not** reproduce verl's 0.0 → 0.715 improvement. The training reward
collapsed to 0.0 (`pass@8` = 0.0) over the final steps, and the run never
finished cleanly (the final HF-weight save OOM'd the node — see Key facts).

| Metric | Start (`eval_before_train`, val) | End |
|---|---|---|
| blended reward (train rollout) | 0.329 | **0.000** (collapsed) |

Training reward (`reward/avg_raw_reward`) over the 15 GRPO steps:

```
0.093 0.092 0.154 0.061 0.011 0.076 0.221 0.181 0.186 0.208 0.248 0.181 0.306 0.000 0.000
```

The curve peaks at 0.306 (step 13) — still below the SFT baseline of 0.329 —
then collapses to 0.0 (degenerate/empty output, `pass@8` = 0.0). This is a
training-dynamics failure (full-model update with weak KL 0.04 forgets the
format), not an infrastructure error.

> **Note on the "before".** Unlike verl, whose `eval_before_train` is 0.0 because
> its thinking-disabled SFT emits empty output, SkyRL's `eval_before_train` was
> **0.329** — the checkpoint emits valid triplets. The two runs' baselines are
> therefore not directly comparable: verl is a "0 → 0.715" learning demo, while
> SkyRL is a "0.329 → 0.0" forgetting failure. (The SkyRL checkpoint appears to be
> a correctly-merged, non-empty variant of the SFT rather than verl's
> thinking-disabled one.)

## Contents

| Path | What it holds |
|---|---|
| `config/SKYRL_CONFIG.md` | every hyperparameter for the RL stage |
| `scripts/env.py` | triplet-F1 reward, SkyRL gymnasium env |
| `scripts/data.py` | verl parquet → SkyRL VLM parquet (256/32) |
| `scripts/entrypoint.py` | SkyRL entrypoint + Qwen3.5 vision-RoPE patch |
| `scripts/run_cholec.sh` | full SkyRL launch command (all CLI overrides) |
| `metrics/` | `final_results.json` + `run_manifest.json` |

## Key facts

- **Full-model RL, not LoRA.** SkyRL FSDP2 trains all parameters (bfloat16) with
  `cpu_offload=true`, so the VLM + fp32 AdamW + master weights fit on 4× A10G
  (24 GB) — but see the OOM note below.
- **Qwen3.5 vision-RoPE patch.** `Qwen3_5VisionRotaryEmbedding` registers
  `inv_freq` as a non-persistent buffer; under `cpu_offload` FSDP leaves it on CPU
  while activations are on cuda, and the vision RoPE forward doesn't move it. The
  `entrypoint.py` patch makes `inv_freq` follow `position_ids.device`.
- **`USE_HUB_KERNELS=NO`.** transformers' precompiled "hub kernels" for the
  Qwen3.5 gated-delta-net layers emit garbage on A10G (sm86); the launch forces the
  fallback to the installed `fla` + `causal_conv1d` packages (the same kernel path
  verl's working run uses).
- **Reward parse.** `env.py` strips leaked `<think>…</think>` blocks and
  `<|…|>` special tokens before parsing triplets.
- **Run did not finish cleanly (OOM).** With `cpu_offload=true`, the final
  `save_hf_model` (consolidating the full model for HF export) pushed RAM to
  185.8 / 186.7 GB and triggered the Ray OOM killer, which infinite-retried. The
  `eval_interval=16` final val eval never ran, and no final weights were written.
  `ckpt_interval=0` + `hf_save_interval=16` was set to avoid the ~53 GB full
  checkpoint (weights + fp32 optimizer) on the small disk; the HF-weights-only
  save is what OOM'd.

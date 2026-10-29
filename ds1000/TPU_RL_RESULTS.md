# DS-1000 RL on Cloud TPU (verl + TorchTitan + vLLM) — result

**Date:** 2026-10-29
**Platform:** GCP Cloud TPU `v6e-8` (8 chips, single host), `verl-cholec-trainer` in `europe-west4-a`
**Stack:** verl (V1 `separate_async` trainer) + `verl-hardware-plugin` (`torchtitan.experiments.tpu.qwen3_5`) + `vllm-torchtpu` rollout + TorchTPU (`torch_tpu`)

## Outcome

10 GRPO steps completed end-to-end with **no fatal errors** in ~7 minutes
(`Training Progress: 100% | 10/10 [07:10]`).

| metric | value |
|---|---|
| model | `Qwen3.5-4B` (text-only) |
| dataset | DS-1000 (800 train / 128 val) |
| steps | 10 |
| wall time | ~7 min (~20–210 s/step, faster after compile) |
| final `critic/score/mean` (DS-1000 pass rate) | **0.234** |
| final `critic/rewards/mean` | 0.234 |

Per-step `critic/score/mean`:

```
0.375  0.375  0.313  0.250  0.297  0.344  0.125  0.094  0.344  0.234
```

Scores fluctuate rather than climb — expected for a 10-step smoke test (16-sample
train batch, `rollout.n=4`), and the point of this run was to prove the
verl + vLLM + TorchTPU pipeline works on Cloud TPU.

## What was required to make it run

`Qwen/Qwen3.5-*` are **multimodal** (`Qwen3_5[Moe]ForConditionalGeneration`), but the
TorchTitan TPU engine drops the vision encoder and trains/serves the **language model
only**. The recipe (`examples/tpu/grpo/run_qwen3_5_torchtitan.sh`) therefore requires a
**text-only** checkpoint (`Qwen3_5[Moe]ForCausalLM`), i.e. the HF checkpoint's language
model with the vision tower removed.

### 1. Text-only checkpoint conversion — `convert_qwen35_text.py`

Renames `model.language_model.*` → `model.*`, drops `model.visual.*` (vision) and
`mtp.*` (MTP heads), keeps `lm_head.weight`, and rewrites `config.json` to
`Qwen3_5ForCausalLM` / `qwen3_5_text` with `text_config` flattened.

```
python convert_qwen35_text.py --src ~/models/Qwen3.5-4B --dst ~/models/Qwen3.5-4B-text
```

**Tied-embedding gotcha:** `Qwen3.5-4B` has `tie_word_embeddings=True` (no separate
`lm_head.weight`), but TorchTitan expects a separate `lm_head`. After conversion, add
`lm_head.weight` as a copy of `model.embed_tokens.weight`. (`9B` has
`tie_word_embeddings=False`, so it does not need this.)

### 2. Trainer / rollout chip contention (single host) — `tpu_vllm_patches.chip_offset.patch`

The plugin assigns both trainer and rollout workers `TPU_VISIBLE_CHIPS = local_rank`
(0..N-1) with no offset, so on a single `v6e-8` host the trainer (chips 0-3) and the
vLLM rollout (also 0-3) collide:

```
RuntimeError: [TorchTPU] Failed to acquire a TPU device node '/dev/vfio/0' ...
             because it is already opened by another process on this host.
```

The README assumes **2 separate slices**. Fix: offset the rollout chip by
`N_CHIPS_TRAINER` (patch in `tpu_vllm_patches.chip_offset.patch`), and export
`TPU_ROLLOUT_CHIP_OFFSET=$((NNODES_TRAINER*N_CHIPS_TRAINER))`.

### 3. Ray `env_vars` must be a string

The Hydra override must quote the value so Ray sees `Dict[str, str]`:

```
+ray_kwargs.ray_init.runtime_env.env_vars.TPU_ROLLOUT_CHIP_OFFSET="'${TPU_ROLLOUT_CHIP_OFFSET}'"
```

(unquoted, Hydra parses it as an int and Ray raises `TypeError`.)

### 4. 9B dense does not fit 4 trainer chips (OOM)

`Qwen3.5-9B` (dense) on 4 chips OOMs: fp32 Adam ≈ 18 GB + weights 4.5 GB + grads
4.5 GB ≈ 27 GB fixed per chip, leaving < 5 GB for the ~5 GB temporaries (`reduce_scatter`,
`call_custom_kernel`, `tt_jit_add`) that every op needs against 32 GB HBM. `optim_impl=fused`,
`use_kl_loss=False`, and shorter seq_len freed only ~2.5 GB — not enough.

**Use `Qwen3.5-4B`** (weights 2 GB + Adam 8 GB + grads 2 GB ≈ 12 GB/chip), which the
recipe explicitly supports on 4 chips.

### 5. Rollout fragmentation (transient, non-fatal)

`vllm-torchtpu` occasionally hits `RuntimeProgramAllocationFailure` while reserving a
~3.5 GB buffer (HBM fragmented by the KV block pool). It self-recovers via compaction.
`gpu_memory_utilization=0.4` (vs default 0.6) avoids it.

## Run script

`run_ds1000.sh` (the standalone-VM wrapper around the official `run_tpu.sh`).

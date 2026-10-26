# verl LoRA GRPO on Cloud TPU (Tinker replication)

## Goal

Replicate Tinker's **LoRA RL stage** — GRPO/RLVR, LoRA r=16, 256 train / 300 val,
triplet-F1 reward — on **verl + a single `v6e-8`**, initialized from the merged SFT
checkpoint. Target: match or beat Tinker's **0.364 → 0.419**.

## Confirmed inputs (verified 2026-10-26)

| Input | State |
|---|---|
| SFT model | merged HF model at `s3://stanford-train-data/models/cholec-sft-qwen3.5-4b` (10.6 GB, 739 tensors) |
| Merge correctness | `tie_word_embeddings: false`, untied `lm_head.weight` — the *fixed* merge; should baseline ~0.364, not 0.0 |
| Tinker LoRA config | r=16, `train_attn` + `train_mlp` + `train_unembed` (untied `lm_head`); base `Qwen/Qwen3.5-4B` |
| Data | CholecT50 — 256 train / 300 val (Tinker manifest splits) |
| Rendering | `qwen3_5_disable_thinking` → plain triplets, no `<think>` |
| Reward | `0.4*triplet_F1 + 0.6*mean(instrument,verb,target F1)` |

## Core design

- Load the merged SFT model as the **frozen base** (encodes the 0.364 behavior).
- Attach a **fresh LoRA r=16** with Tinker's targets (`q/k/v/o` + `gate/up/down` + **`lm_head`**).
- Reference model (KL 0.04) = the merged SFT base (frozen). With zero-init LoRA,
  initial policy = base, so `val_before_train` **is** the baseline check.
- Tinker `importance_sampling` ≈ verl `adv_estimator=grpo` (group-normalized advantages).

## Reused from `tpu_rl_benchmark` (SkyRL TPU)

`rollout/` (HF engine + `tpu_compat.py` + supervisor), `tpu_groups.py`,
`device.py` shim pattern, `/dev/shm` reload weight-sync, `ops/single_vm.sh` +
`gcloud/*`, `config/tpu.env.sh` (renamed VMs).

## verl-specific work

1. `verl_tpu/device.py` — device shim (TorchTPU registration, `torch.accelerator`,
   FSDP `_move_states_to_device` + mRoPE patches).
2. Rollout worker — replace verl's in-process vLLM with a worker fronting the HF engine.
3. FSDP1 vs FSDP2 — gating probe (SkyRL only proved FSDP2 on TorchTPU).
4. LoRA specifics:
   - targets must include **`lm_head`** (Tinker `train_unembed=True`).
   - weight-sync must export **base + LoRA merged** (incl. untied head) to rollout.
5. Parity checker — verl Hydra keys vs `VERL_CONFIG.md`, LoRA additions documented.

## Phases (probe-gated)

| Phase | Work | Gate |
|---|---|---|
| 0 | Clone verl, pin `uv.cu130`, confirm LoRA RL support + rollout sync | LoRA feasible |
| 1 | Scaffold `~/verl-tpu/`, write `device.py` + setup + parity | imports pass |
| 2 | Provision `v6e-8`, build TorchTPU + verl env, fetch S3 ckpt + parquet | device check |
| 3 | P0 (device/FSDP) · P1 = baseline check on TPU (greedy 300-val → ~0.364) · P2 (LoRA GRPO grad TPU vs CPU) | P0+P1 pass; P2 agrees |
| 4 | Full 16-step LoRA GRPO, parity-gated, self-running spot job | completes |
| 5 | Compare vs Tinker 0.364 → 0.419 | decision |

## Success

- ✅ reach ≥ 0.419 from ~0.364 start (reproduces Tinker's +15%).
- ⚠️ starts ~0.0 → merge is wrong; stop at P1.
- ❌ collapses like SkyRL full-model → P2 rules TPU-gradient in/out.

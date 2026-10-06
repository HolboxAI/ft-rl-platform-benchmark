# Experiment Results — Prime Intellect (Fireworks-parity, 4 experiments)

**Last updated:** 2026-10-06
**Status:** 2 of 4 experiments complete (GoEmotions SFT + RL); **both** ORena VLM legs blocked by pod-image dependency drift (SFT: VLM-SFT validator; RL: TileLang/CCCL/CUDA kernel-JIT mismatch).

---

## 0. How to read this

The 4 reference experiments are reproduced on Prime Intellect via **on-demand pods** (`prime pods` → SSH → self-run `prime-rl`), *not* the hosted "training runs/jobs" surface. So results here come from pod logs, not the dashboard. Parity rule (from `DECISIONS_LOG.md` D6): every reference hyperparameter is kept identical; we shrink **only** data volume and step count to fit the 20–30 min wall.

---

## 1. Status at a glance

| Exp | Task | Model | Method | Status | Final metric |
|---|---|---|---|---|---|
| 1 | ORena SFT | Qwen3.5-9B (VLM) | LoRA r=64 | 🚫 **BLOCKED** (VLM SFT version drift) | — |
| 2 | ORena GRPO | Qwen3.5-9B (VLM) | LoRA r=16 | 🚫 **BLOCKED** (TileLang/CCCL/CUDA image drift) | — |
| 3 | GoEmotions SFT | Qwen3-0.6B | LoRA r=16 | ✅ **COMPLETE** | micro-F1 **0.4956** / Jaccard 0.3294 |
| 4 | GoEmotions GRPO | Qwen3-0.6B | LoRA r=16 | ✅ **COMPLETE** | micro-F1 **0.0349** / Jaccard 0.0178 |

---

## 2. Exp 3 — GoEmotions SFT (✅ COMPLETE)

**Config** (`configs/sft_goemotions.toml`):
- model `Qwen/Qwen3-0.6B`; LoRA r=16 α=32 dropout=0.05 (10,092,544 adapter params over 440.4M base)
- data: 800 train rows (`/workspace/data/goemotions/chat`), seq_len 256, batch 64 / micro 8, loss_mask = assistant-only
- optim adamw lr 1e-4; scheduler cosine warmup 10; max_steps 100
- renderer auto → **Qwen3Renderer** (detected automatically)
- `[ckpt.weights] save_adapter_separately = true` (see bug note below)

**Final loss curve (re-run):**

| Step | Loss |
|---|---|
| 1 | 5.7935 |
| 6 | 3.2676 |
| 25 | 0.4760 |
| 50 | 0.2648 |
| 75 | 0.2084 |
| 100 | **0.1672** |

- Loss fell **5.79 → 0.167** over 100 steps (real learning). Throughput ~2,800 tok/s; peak VRAM **3.3 / 39.5 GiB**.

**Final held-out metrics (200-row dev set, greedy decode):**

| Metric | Value |
|---|---|
| micro-F1 | **0.4956** |
| micro-Jaccard | 0.3294 |
| mean instance-F1 | 0.4948 |
| mean instance-Jaccard | 0.4708 |

*Greedy decode with `enable_thinking=False`; labels parsed from comma-separated output; TP/FP/FN = 113/104/126.*

**⚠️ Platform bug found & worked around:** prime-rl's default `save_adapter_separately = false` **silently drops the LoRA adapter** at checkpoint time — `clean_lora_state_dict` deletes `lora_A`/`lora_B` and keeps only the frozen base weights (despite the docstring claiming "before merging into full model weights"). Result: the saved `model.safetensors` is the *un-fine-tuned base model*, and a naive eval scores micro-F1 ≈ **0.02** (base-model behavior). Fix: re-run with `save_adapter_separately = true`, then merge `lora_adapters/adapter_model.safetensors` into the base (`W' = W + (B@A)·α/r`, α/r = 2.0). After merge, micro-F1 = **0.4956**.

---

## 3. Exp 4 — GoEmotions GRPO (✅ COMPLETE)

**Config** (`configs/rl_goemotions.toml`):
- model `Qwen/Qwen3-0.6B`; LoRA r=16 α=32 dropout=0.05; optim lr 1e-5
- 800 prompts, batch 64 / group 8 (8 rollouts/prompt), `max_steps = 64`
- sampling temperature 0.7, `max_completion_tokens = 64`
- reward = multi-label Jaccard (taskset `goemotions_v1`)
- `[orchestrator.renderer] name="qwen3" enable_thinking=false`
- `[inference] gpu_memory_utilization=0.6` + `[inference.model] max_model_len=512`
- `save_adapter_separately=true`

**Final held-out metrics (200-row dev set, greedy decode, thinking disabled):**

| Metric | Value |
|---|---|
| micro-F1 | **0.0349** |
| micro-Jaccard | 0.0178 |

**Training reward curve (`jaccard_curve`, per-step mean reward):** flat/noisy, range 0–0.073,
no upward trend. A fresh `Qwen3-0.6B` base model + Jaccard reward over 64 steps is not enough
signal to move the label-set prediction — versus Fireworks' 0.175 → 0.377 (micro-F1) on 5000
prompts.

**Three launch blockers fixed before the run completed:**
1. **Thinking mode** — Qwen3's renderer opens a `<think>` block by default, so a fresh model
   spends the whole 64-token budget "reasoning" and never emits the answer (Truncation 100%,
   reward 0). `enable_thinking=false` dropped truncation to ~0-3%.
2. **zero-advantage filter** — GRPO's default `post_batch_filters` enforce `zero_advantage`,
   which drops every group whose rollouts share a reward (the common case for a fresh model)
   and aborts with "10 consecutive empty train batches". Overriding to `enforce=false` lets
   training proceed.
3. **Single-GPU vLLM memory** — the trainer and inference server share physical GPU 0
   (`CUDA_VISIBLE_DEVICES="0,0"` hack to satisfy the disjoint-GPU check); vLLM's default
   `gpu_memory_utilization=0.9` fails because the trainer already holds ~3.8 GiB. Lowered to
   0.6 and capped `max_model_len=512`.

---

## 4. Exp 1 — ORena SFT (🚫 BLOCKED — VLM SFT version drift)

The pod image ships **prime-rl 0.7.0**, whose SFT data path is renderer-only and rejects
`[model.vlm]`:

> `validate_renderer_vs_vlm: "renderer-only SFT does not support VLMs yet. The renderer
> tokenizes text-only message dicts client-side and cannot handle image inputs."`

GitHub `main` has **removed** this validator (its only remaining VLM/SFT constraint is
`freeze_vision_encoder=false` being LoRA-incompatible), so ORena SFT is blocked by the **pod
image version**, not by the platform itself. This is the mirror of Fireworks' gap: Fireworks
cannot RL `qwen3p5-9b`; Prime Intellect 0.7.0 cannot SFT a VLM but **can** RL one (the RL
config carries `vlm` and rolls out through vLLM, which handles images natively). A prime-rl
upgrade on the pod was judged too risky; recorded as a version-drift finding instead.

---

## 5. Exp 2 — ORena GRPO (🚫 BLOCKED — TileLang / CCCL / CUDA image drift)

**Config** (`configs/rl_orena.toml`):
- model `Qwen/Qwen3.5-9B` (VLM); LoRA r=16 α=32 dropout=0.05; optim lr 1e-5
- 128 prompts, batch 64 / group 8 (8 rollouts/prompt), `max_steps = 48`
- sampling temperature 0.7, `max_completion_tokens = 512`
- reward = exact match after normalization (taskset `orena_v1`)
- `[orchestrator.renderer] name="qwen3.5" enable_thinking=false`
- `[trainer.model] impl="custom"` + `optimization_dtype/reduce_dtype="bfloat16"` (VLM requirement)
- `[inference] gpu_memory_utilization=0.6`; `[inference.model] max_model_len=8192`
- `save_adapter_separately=true`

**Outcome:** the trainer crashed **three times**, each a deeper layer of the same broken
dependency chain in the `prime_rl 0.7.0` image. Qwen3.5-9B's architecture exercises the `fla`
(flash-linear-attention) path, whose kernels are JIT-compiled during the trainer's backward
pass via **TileLang 0.1.9** — a path the text-only Qwen3-0.6B (Exps 3–4) never touches:

1. `cuda/atomic: No such file` → fixed with `CPLUS_INCLUDE_PATH` pointing at the shipped CCCL.
2. `cccl/cuda/std/utility: No such file` → the bundled cutlass expects the **CUDA 13 CCCL
   `cccl/` layout**; the image ships CUDA 12.9's `cuda/` layout → symlinked `cccl/*→*`.
3. `cuModuleLoadData … CUDA_ERROR_INVALID_IMAGE` → the kernel now *compiled* but produced a
   cubin the driver rejects.

**Root cause:** the image carries a stray **CUDA 13 nvcc** (`nvidia/cu13` 13.2.78) that
TileLang discovers, mismatched with the CUDA 12.8 driver + CUDA 12.9 CCCL;
`nvidia-cuda-cccl-cu13` is only a `0.0.1` stub, so the toolchain cannot be made self-consistent
in place. Two steps *did* train (reward 0.09 → 0.23, `Error 0%`, `Truncation 0%`), so the RL
method itself is sound on this model — the image simply can't compile the linear-attention
kernels. Mirror image of Exp 1: the pod image runs **neither** ORena VLM leg.

---

## 6. Final results (parity table)

| Exp | Metric | Fireworks (ref) | Prime Intellect | Δ |
|---|---|---|---|---|
| 1 ORena SFT | meanAccuracy (4-bucket) | — | 🚫 blocked (VLM SFT version drift) | — |
| 2 ORena RL | meanAccuracy (4-bucket) | — | 🚫 blocked (TileLang/CCCL/CUDA image drift) | — |
| 3 GoEmotions SFT | micro-F1 | 0.313 (train loss; no F1 reported) | **0.4956** (train loss 0.167) | not like-for-like (800 vs 43k samples) |
| 4 GoEmotions RL | micro-F1 / Jaccard | 0.175 → 0.377 (F1) | **0.0349** / 0.0178 | fresh 0.6B + Jaccard over 64 steps ≈ no signal |

---

## 7. Billing / resource visibility (headless)

- **Balance:** **−$87.10 USD** (wallet overdrawn; team "Holbox AI"). All pods terminated — no
  further billing accrues.
- **Pod spend (final, per `prime wallet`):**
  | Pod | GPU | Rate | Used for | Spend |
  |---|---|---|---|---|
  | 484b6053… | A10 24GB | ~$1.24/hr | Exp 4 RL (+ failed launches) | $6.01 |
  | 19b97318… | H200 141GB | $4.50/hr | ORena SFT attempt, **idle** (blocked) | $16.15 |
  | cd5f26c0… | H200 141GB | $4.50/hr | ORena RL attempt, **idle** (blocked) | **$114.94** |

  Total ≈ **$137.10**. The two H200 pods were the overwhelming cost and produced **no run** —
  a direct consequence of the VLM blocks being discovered *after* provisioning, with the
  `cd5f26c0…` pod sitting idle overnight while the TileLang/CCCL/CUDA crash chain was
  debugged. This is the lesson to carry forward: confirm the image can actually run a VLM
  forward+backward on a cheap text-only pod *before* paying for an H200.
- **Why the web UI shows nothing:** these are on-demand **pods**, not hosted **training runs**
  (`prime train list` = empty). The pods/credits are visible via `prime pods list` and
  `prime wallet`; they may not be surfaced in the dashboard's "Jobs" view.

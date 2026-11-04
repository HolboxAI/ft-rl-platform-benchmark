# Trainium2 — platform-wise blocker list

Consolidated, in-depth record of every thing that blocks a **successful CholecT50
GRPO/RLVR run on Trainium2 (TorchNeuron)**, organised per framework, based on our
experiments across five platforms: **TRL, verl, SkyRL, Miles, TitanRL**.

Each per-framework list lives in its own issues log. This file is the
cross-platform view: what is shared, what is unique, and what one fix would
unblock several platforms at once.

Source of truth, per leg:

- TRL — `TRAINIUM_ISSUES_LOG.md` (23 issues, empirically run to a full epoch)
- verl — `verl/VERL_TRAINIUM_ISSUES_LOG.md` (6 issues, source + on-instance gate)
- SkyRL — `skyrl/SKYRL_TRAINIUM_ISSUES_LOG.md` (5 issues) + `SkyRL` source
- Miles — `miles/MILES_TRAINIUM_ISSUES_LOG.md` (5 issues)
- TitanRL — `titanrl/TITANRL_TRAINIUM_ISSUES_LOG.md` (6 issues, on-instance probe)

> **Update 2026-11-04.** Source analysis of `vllm@5225a1f` vs
> `vllm-neuron@a080ebf` refines the C4/C5 fix. Upstream vLLM already has
> Qwen3.5-VL + full GatedDeltaNet (CUDA/HIP/CPU) + the RL weight-sync API
> (`sleep`/`wake_up`/`update_weights`/`update_weight_version`) + a pluggable
> weight-transfer framework. vLLM-neuron is missing only: Qwen3.5 hybrid-GDN,
> a neuron weight-transfer backend, and neuron `sleep`/`wake_up`. The whole
> rollout+weight-sync wall collapses to that bounded contribution — see
> `verl/IMPLEMENTATION_PLAN.md`. Miles still needs SGLang, not vLLM.

## Evidence legend

| Tag | Meaning |
|---|---|
| `CONFIRMED` | hit on a live `trn2.48xlarge` with exact error text |
| `SOURCE` | source-verified (file:line), no instance needed |
| `CARRIED` | expected model-level wall; would hit after the framework issues clear |
| `OPEN` | no workaround yet |
| `WORKAROUND` | in-repo code patch exists, upstream fix still wanted |
| `NATIVE-FIX` | needs an upstream (package) change |

---

## Summary table

| Framework | Out-of-box on Trainium2 | Rollout engine | Weight-sync | Top blocker |
|---|---|---|---|---|
| **TRL** | **RUNS** (single-core LoRA) | none (in-process `model.generate`) | none (single process) | model-level only (GDN, TDRAM, single-core memory) |
| **verl** | No | vLLM/SGLang async (CUDA) | NCCL/HCCL/Gloo | **rollout async-server-only** |
| **SkyRL** | No (runs only via our port) | vLLM (CUDA) | NCCL/IPC (`delta` alt) | CUDA device layer + vLLM rollout |
| **Miles** | No | SGLang (CUDA) | SGLang reload + `torch.distributed` | SGLang-only rollout + no neuron platform |
| **TitanRL** | No | vLLM (CUDA, gated) | TorchStore (portable) | deps not on PyPI + vLLM rollout + VLM text-only |

---

## Cross-cutting blockers (shared; block 3–5 platforms each)

These are the highest-value fixes. They live in the **model** or the **Neuron
toolchain**, not in any one framework, so one fix unblocks several legs.

### C1 — Gated DeltaNet has no Neuron kernel (blocks all 5)

- **Evidence**: `CONFIRMED` — TRL `TRAINIUM_ISSUES_LOG.md` #11.
- **What**: `causal_conv1d`, `flash-linear-attention`, `chunk_gated_delta_rule`,
  `causal_conv1d_update`, `fused_recurrent_gated_delta_rule` are all CUDA-only.
  Qwen3.5 has 24 GatedDeltaNet layers of 32, so 24 layers fall back to reference
  PyTorch.
- **Effect**: correctness is fine, throughput is not — an order-of-magnitude
  (10–50×) slowdown. This is the deepest single blocker: it is in the model, so
  it hits TRL, verl, SkyRL, Miles, and TitanRL equally.
- **Fix**: ship `causal_conv1d` + `flash-linear-attention` on the Neuron AMI, or a
  Neuron-equivalent fused GDN kernel.

### C2 — Static shapes / on-core TDRAM exhaustion (blocks all 5)

- **Evidence**: `CONFIRMED` — TRL `TRAINIUM_ISSUES_LOG.md` #23 (validated fix).
- **What**: eager mode compiles one NEFF per unique attention/SDPA shape. Variable
  RL batch lengths (image resolutions + variable completion lengths) create ~40
  shapes, which exhaust the small on-core TDRAM DMA pool around step 40
  (`err: 4` `NERR_RESOURCE`, "Failed to load model").
- **Effect**: mid-training death, not slow-down. Any framework doing variable-length
  RLVR on neuron will hit it.
- **Fix**: torch_neuronx eager should evict/reuse NEFFs (and their TDRAM staging)
  when the pool fills, or fail with an actionable "TDRAM exhausted (N resident
  NEFFs)". Frameworks should also expose fixed-length completion padding.

### C3 — Dynamic `torch.repeat_interleave` in the vision tower (blocks all 5 VLM legs)

- **Evidence**: `CONFIRMED` — TRL `TRAINIUM_ISSUES_LOG.md` #1.
- **What**: `get_vision_interpolation_indices_and_weights` computes
  `repeat_interleave(grid_thw[:, 1], counts)` with a tensor-valued `repeats`.
  neuronx-cc cannot lower it (`BIRCodeGenLoop error: tuple index out of range`).
- **Effect**: compile crash (or silent CPU fallback in eager). All legs use
  Qwen3.5-VL, so all hit it.
- **Fix**: neuronx-cc should lower dynamic `repeat_interleave`, or fall back to CPU
  under `torch.compile`. Workaround (compute indices on CPU) already patched in
  TRL + SkyRL + the verl plugin.

### C4 — Rollout engine is CUDA-only (blocks verl, SkyRL, Miles, TitanRL)

- **Evidence**: `SOURCE` + `CONFIRMED` (TitanRL probe: vLLM `DeviceConfig("auto")`
  → `RuntimeError: Failed to infer device type`, `current_platform =
  UnspecifiedPlatform`).
- **What**: the rollout (generation) side is vLLM or SGLang everywhere except TRL.
  No neuron rollout server exists as a first-party, framework-agnostic component.
- **Effect**: the biggest single framework-level wall. TRL avoids it because it is
  single-process (no separate rollout). The other four cannot start generation.
- **Fix**: a neuron rollout with paged attention + continuous batching — ideally
  **vLLM-neuron** (AWS beta) serving Qwen3.5-VL, or a purpose-built neuron
  generate server. See "Shared resolve path" below.

### C5 — No neuron collective for full-weight sync (blocks verl, SkyRL, Miles)

- **Evidence**: `SOURCE` — verl collective backends are `nccl`/`hccl`/`gloo`;
  SkyRL `weight_sync_backend="nccl"` (colocated → CUDA IPC); Miles SGLang reload
  via `torch.distributed`.
- **What**: full-model weight sync between trainer and rollout defaults to
  NCCL/CUDA-IPC. (TitanRL is the exception — TorchStore is portable — but its
  deps are not installable, see its list.)
- **Effect**: full-fine-tune sync either fails or becomes a CPU round-trip every
  step. LoRA-only sync (what our SkyRL port does) sidesteps it but does not scale.
- **Fix**: a neuron collective (neuronx-distributed / gloo mesh) for FSDP + weight
  sync. We already prototyped the gloo→neuron mesh in SkyRL.

### C6 — neuronx-cc `masked_scatter` instruction limit (blocks all 5 at scale)

- **Evidence**: `CONFIRMED` — TRL `TRAINIUM_ISSUES_LOG.md` #20.
- **What**: the response-only loss mask (`masked_scatter` over `[16, 640, 2560]`)
  unrolls to 131 M instructions, ~437× the 300 k "typical limit"
  (`NCC_EXTP003`).
- **Effect**: full-size single-core batches fail to compile. Forces small batches
  (more steps) or tensor parallelism.
- **Fix**: neuronx-cc should lower large `masked_scatter` as gather/reindex, or
  frameworks should emit a fused masked-logprob kernel.

### C7 — OOM / trace-cache ergonomics (blocks all 5, tooling)

- **Evidence**: `CONFIRMED` — TRL `TRAINIUM_ISSUES_LOG.md` #8, #9, #10.
- **What**: (a) OOM recovery loops forever instead of raising; (b) the NKI trace
  cache pins ~10 GB HBM by default and is not reclaimed on OOM; (c) the OOM
  message is generic ("Failed to allocate resource").
- **Effect**: hung processes, wasted HBM, hard to debug.
- **Fix**: raise `torch.OutOfMemoryError` after N attempts; make the trace cache
  reclaimable; clearer OOM messages.

---

## Platform-wise lists

### TRL — RUNS (single-core, LoRA) — the neuron-native baseline

TRL is the only framework that runs out of the box. Its blockers are
**model-level**, not framework-level: the things that stop a *comparable /
production* run (full fine-tune, multi-core, `beta=0.04`), not a first run.

**What stops production:**

1. **Single-core HBM ceiling** — full fine-tune fp32 AdamW states (~40 GB) and the
   KL reference model (~10 GB) do not fit 25.8 GB HBM. Forced LoRA r=16 +
   `beta=0.0`. → needs multi-core TP (NeuronX Distributed) or a CPU-offload
   optimizer. `OPEN` (#8).
2. **C1 GDN fallback** — 24/32 layers slow. `OPEN` (#11).
3. **C2 TDRAM exhaustion** — fixed via constant forward shape, but the fix
   restricts batch/length flexibility. `WORKAROUND` (#23).
4. **C3 `repeat_interleave`** — patched. `WORKAROUND` (#1).
5. **C6 `masked_scatter`** — forces small batches. `OPEN` (#20).
6. **C7 OOM/trace-cache** — trace-cache pin works around by
   `NKI_ENABLE_TRACE_CACHE=0`. `WORKAROUND`/`NATIVE-FIX` (#9/#10).
7. **Checkpoint saving** — the completed epoch used `save_strategy="no"`; the
   trained adapter was discarded. Process gap, not a library issue.

### verl — NOT runnable out-of-box (6 framework issues)

**Framework issues (in the order an instance hits them):**

1. **Python `<3.13` cap** (`verl 0.9.1`) vs DLAMI 3.14.4. `RESOLVED` (use 0.9.0).
2. **`tensordict` pin has no 3.14 wheels**. `RESOLVED` (0.15.3 override).
3. **`transformers` cap conflicts with `qwen3_5`'s 5.18.0**. `WORKAROUND` (keep 5.18).
4. **`orjson` imported but not declared**. `RESOLVED` (manual install).
5. **Device module resolves to `torch_neuronx.neuron`** (not `torch.neuron`).
   `RESOLVED` (in our `PlatformNeuron`).
6. **Rollout is async-server-only** — `get_rollout_class("hf"/"naive")` asserts;
   `_ROLLOUT_REGISTRY` only maps `vllm`/`sglang`/`trtllm` (all CUDA).
   `HFRollout`/`NaiveRollout` are dead code. **This is the real blocker.**
   `BLOCKED` (#6).

**Carried model-level walls** (hit after the rollout exists): C1, C2, C3.

### SkyRL — NOT out-of-box; our port is a PoC

**Framework issues (source-verified):**

1. **`fsdp`/`megatron` extras pin CUDA torch + vLLM** (`torch==2.13.0`,
   `vllm==0.31.0`, `nixl-cu13`). `BLOCKED` (#1).
2. **No device abstraction** — 34 files hardcode `torch.cuda`. `BLOCKED` (#2).
3. **vLLM-only rollout** (`backend="vllm"`). `BLOCKED` (#3).
4. **Weight-sync default `nccl`/IPC** (`delta`/`sharded_rdt` alts exist).
   `PARTIAL` (#4).
5. **`jax` backend has no Qwen3.5 VLM** and pins CUDA/TPU JAX. `NOT A PATH` (#5).

**Our port's gaps (from `neuron_engine.py`, the reason it is not production):**

6. **Single-core, serialized** — `get_world_size()` returns `(1, 1)`; one
   `asyncio.Lock` serializes every forward. ~1/48 of the box, zero batch
   concurrency.
7. **No continuous batching / paged attention** — transformers `model.generate`
   (eager), right-padded per batch.
8. **LoRA-only weight sync** — `update_named_weights` raises
   `NotImplementedError` for full-weight sync.
9. **Per-token CPU log-softmax** — `_response_logprobs` moves logits `.to("cpu")`
   every step (neuronx-cc cannot lower it under NKI).

### Miles — NOT runnable (5 framework issues)

1. **SGLang-only rollout** (no vLLM path, no transformers-native path wired).
   `BLOCKED` (#1). Note: SGLang, not vLLM — so a vLLM-neuron fix does **not**
   cover Miles directly.
2. **FSDP backend hardcodes `torch.cuda`** (~10 sites). `PARTIAL` (#2).
3. **No `neuron` platform** (`MILES_ACCELERATOR` only `cuda`/`tpu`). `BLOCKED` (#3).
4. **Megatron backend + CUDA kernels are NVIDIA-only**. `EXCLUDE` (#4).
5. **Docker bakes CUDA torch + SGLang + Megatron**. `BLOCKED` (#5).

### TitanRL — NOT runnable (6 issues, empirically probed)

1. **`torchstore`/`torchmonarch` not on PyPI** — git-only Meta packages; PyPI
   names are unrelated 0.1.x. `pip install` → `ResolutionImpossible` / "No
   matching distribution". `CONFIRMED` (#1).
2. **vLLM-only rollout + `has_cuda_capability(9,0)` gate** — gate returns `False`
   on neuron; `DeviceConfig("auto")` → "Failed to infer device type".
   `CONFIRMED` (#2).
3. **`pip install vllm` pulls CUDA wheels** (cuda-toolkit 13.0.3,
   nvidia-nccl-cu13, flashinfer). `CONFIRMED` (#3).
4. **`flash-attn` source build fails** (no `torch` in build env; CUDA kernel).
   `CONFIRMED` (#4).
5. **`attn-gym[cudnn]` resolves to NVIDIA cutlass/cuda-toolkit**; GDN is a CUDA
   custom op. `CONFIRMED` (#5).
6. **VLM is "text only"** — rollout takes `prompt_token_ids` (tokens, no image
   modality), so CholecT50 image frames are not expressible even on NVIDIA.
   `SOURCE` (#6).

---

## Shared resolve path (one fix → many platforms)

| Fix | Where it lives | Platforms unblocked |
|---|---|---|
| **Neuron GDN kernel** (C1) | torch-neuronx / neuronx-cc | 5 of 5 |
| **neuronx-cc dynamic `repeat_interleave`** (C3) | neuronx-cc | 5 of 5 (VLM) |
| **NEFF eviction / TDRAM** (C2) | torch-neuronx | 5 of 5 |
| **`masked_scatter` lowering** (C6) | neuronx-cc | 5 of 5 |
| **OOM/trace-cache ergonomics** (C7) | torch-neuronx | 5 of 5 |
| **vLLM-neuron Qwen3.5-VL rollout** (C4) | vLLM-neuron (AWS) | verl, SkyRL, TitanRL |
| **SGLang-neuron** (or a vLLM path) (C4) | SGLang / Miles | Miles |
| **Neuron collective** (C5) | neuronx-distributed | verl, SkyRL, Miles |

## Priority order (cheapest decisive signal first)

1. **Verify vLLM-neuron serves Qwen3.5-VL** (vision + GDN) on a trn2. One-day
   probe; the go/no-go for C4. If it works, the rollout is solved for 3 of 4
   distributed frameworks.
2. **Land the GDN kernel** (C1). Deepest shared blocker; blocks all 5.
3. **neuronx-cc fixes** (C3, C6) + **TDRAM eviction** (C2). Unlock multi-core and
   larger batches.
4. **Neuron collective** (C5). Full-weight sync at scale.
5. **Thin per-framework rollout adapters** (verl, SkyRL, TitanRL, Miles) + the
   already-written trainer ports.

## Bottom line

Only **TRL** runs on Trainium2 out of the box, and only in single-core LoRA mode.
The four distributed frameworks (verl, SkyRL, Miles, TitanRL) all fail on the
same two walls — a **CUDA-only rollout** and (for three of four) a **CUDA-only
weight sync** — layered on top of four shared **model/toolchain** walls (GDN,
TDRAM, `repeat_interleave`, `masked_scatter`). One shared investment — a neuron
GDN kernel + a vLLM-neuron rollout + a neuron collective — removes the wall for
most of them at once. The GDN kernel is the single deepest blocker.

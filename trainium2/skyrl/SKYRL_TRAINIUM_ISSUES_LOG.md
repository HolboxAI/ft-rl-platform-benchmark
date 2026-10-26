# SkyRL on Trainium2 (TorchNeuron) — issues log

Same purpose as the TRL and verl issue logs. This one is **source-verified**
against `NovaSky-AI/SkyRL` @ `69410e7` (main): no Trainium instance was
launched, because SkyRL's incompatibility is fully determined by its source and
dependency pins. Issue numbers are stable references for the package-changes
list at the bottom.

---

## Milestones

- **2026-10-26** — cloned SkyRL main (`69410e7`) and inspected `skyrl/`
  (train, backends, tinker, tx). Verdict: CUDA-hardcoded device layer + vLLM-only
  rollout + CUDA dependency pin. Source-verified, no instance.

---

## What is true (source-verified)

1. `skyrl-train` is a separate codebase from verl (not built on it), with three
   training backends (`fsdp`, `megatron`, `jax`).
2. The CholecT50 leg runs `fsdp` + vLLM on CUDA.
3. 34 files touch `torch.cuda`; there is no device abstraction.
4. The only inference backend is vLLM; anything else raises.

---

## Issues (in the order an instance would hit them)

### #1 — `fsdp`/`megatron` extras pin CUDA torch + vLLM (dependency wall)

- **Status**: BLOCKED (source-verified).
- **Class**: package-side (SkyRL packaging).
- **Observed** (pyproject.toml): `fsdp` → `torch==2.13.0`, `vllm==0.31.0`,
  `nixl-cu13`; `megatron` → same + `transformer-engine`, `flash-mla +cu13`.
- **On Trainium**: installing `skyrl[fsdp]` installs CUDA torch, clobbering
  torch-neuronx (the only torch that runs on Trainium). No `neuron` extra.
- **Fix**: make the torch/vLLM pins environment-aware; add a `neuron` extra.

### #2 — No device abstraction; 34 files hardcode `torch.cuda`

- **Status**: BLOCKED (source-verified).
- **Class**: package-side (SkyRL device layer).
- **Observed**: `distributed/fsdp_strategy.py` (`torch.cuda.set_device`,
  `current_device`), `distributed/fsdp_utils.py`, `train/utils/utils.py`
  (`torch.cuda.is_available()` P2P gate),
  `megatron/fused_linear_logprob_triton.py` (`get_device_capability`).
- **On Trainium**: `torch.cuda.set_device(0)` raises (no CUDA device). verl
  0.9.0 solved this with `PlatformBase`; SkyRL has no equivalent.
- **Fix**: introduce a device abstraction (or patch the 34 sites) that resolves
  `neuron`; make Triton capability dispatch + P2P + FSDP device selection
  device-aware.

### #3 — Rollout is vLLM-only (no transformers-native engine)

- **Status**: BLOCKED (source-verified).
- **Class**: package-side (SkyRL inference engine).
- **Observed**: `InferenceEngineConfig.backend: str = "vllm"`;
  `get_sampling_params_for_backend` raises
  `Unsupported generation backend: {backend}` for non-vllm. SGLang dropped in
  v0.1.0.
- **On Trainium**: no vLLM → no rollout. There is no `HFRollout` equivalent.
- **Fix**: add a transformers-native inference engine (`model.generate()` on
  `neuron`) behind `InferenceEngineInterface`, selectable via `backend="neuron"`.

### #4 — Weight-sync default is CUDA (`nccl`/`ipc`)

- **Status**: PARTIAL (a CUDA-free path exists but is not the default).
- **Class**: package-side (SkyRL weight sync).
- **Observed**: `weight_sync_backend: str = "nccl"`; colocated nccl → CUDA IPC.
  `delta` (checkpoint deltas via disk/gs/s3) and `sharded_rdt` (NIXL) exist.
- **On Trainium**: `nccl`/`ipc` fail; `delta` would work but is moot without a
  rollout.
- **Fix**: add a neuron collective, or default non-colocated runs to `delta`.

### #5 — `jax` backend has no Qwen3.5 VLM and pins CUDA/TPU JAX

- **Status**: NOT A PATH (as-is).
- **Class**: package-side (SkyRL-tx).
- **Observed**: `skyrl/tx/models/` has `llama3` + RoPE only; `jax` extra pulls
  `jax[cuda13]` + `jax[tpu]`.
- **On Trainium**: JAX-XLA could target Trainium, but there is no Qwen3.5 VLM
  model and no neuron JAX extra.
- **Fix**: add a Qwen3.5 VLM to `skyrl/tx/models/` and a `jax[neuron]` extra
  (a large, separate job).

---

## Model-level walls (expected, carried from the TRL/verl legs)

Once #1–#4 are fixed and the loop trains on `neuron`, the same torch-neuronx /
neuronx-cc walls hit:

- Static-shape/TDRAM exhaustion from variable RL batch lengths (#23).
- Gated DeltaNet falls back to reference PyTorch without `causal_conv1d` /
  `flash-linear-attention` (#11).
- Dynamic `repeat_interleave` in the vision tower (#1).

---

## Package-side changes we're waiting on (SkyRL leg)

### skyrl (framework) — 5

- **#1** — `fsdp`/`megatron` extras pin `torch==2.13.0` + `vllm==0.31.0`
  (CUDA). Fix: unpin; add a `neuron` extra.
- **#2** — no device abstraction; 34 files hardcode `torch.cuda`. Fix: add a
  device layer (or patch the sites) that resolves `neuron`; make Triton
  capability / P2P / FSDP device-aware.
- **#3** — vLLM-only rollout (`backend="vllm"`). Fix: add a transformers-native
  inference engine on `neuron`.
- **#4** — weight-sync default `nccl`/`ipc` is CUDA. Fix: add a neuron
  collective or default to the CUDA-free `delta` path.
- **#5** — `jax` backend has no Qwen3.5 VLM and pins CUDA/TPU JAX. Fix: add the
  VLM + a `jax[neuron]` extra.

### Carried over from the TRL leg (same runtime/compiler stack) — 7

- **torch-neuronx** — #8 unclear OOM, #9 OOM recovery loop, #10 trace-cache
  pin, #11 GDN kernel fallback, #23 TDRAM exhaustion.
- **neuronx-cc** — #1 dynamic `repeat_interleave`, #20 `masked_scatter`
  instruction limit.

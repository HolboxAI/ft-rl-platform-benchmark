# Miles on Trainium2 (TorchNeuron) — issues log

Same purpose as the TRL, verl, and SkyRL issue logs. This one is
**source-verified** against `radixark/miles` @ `ba60431` (main): no Trainium
instance was launched. Issue numbers are stable references for the
package-changes list at the bottom.

---

## Milestones

- **2026-10-26** — cloned `radixark/miles` (`ba60431`) and inspected `miles/`
  (platforms, rollout, backends, kernels). Verdict: closest of the three RL
  frameworks to native — has a device abstraction with a TPU precedent and a
  custom-generate-function rollout hatch. Source-verified, no instance.

---

## What is true (source-verified)

1. Miles is a fork of `slime`; rollout = SGLang, training = Megatron / FSDP /
   torchtitan.
2. `miles/platforms/` has a `Platform` base + `TpuPlatform`, selected by
   `MILES_ACCELERATOR=cuda|tpu`, using `torch.accelerator` for device ops.
3. The rollout has a documented `--custom-generate-function-path` /
   `--rollout-function-path` escape hatch.
4. 26 files touch `torch.cuda`; the FSDP backend alone has ~10 direct sites.

---

## Issues (in the order an instance would hit them)

### #1 — Rollout is SGLang-only (CUDA); no transformers-native path wired

- **Status**: BLOCKED (source-verified); low friction to fix.
- **Class**: package-side (Miles rollout).
- **Observed**: default `generate` POSTs to
  `http://{sglang_router_ip}:{sglang_router_port}/generate`. SGLang is CUDA/ROCm
  only.
- **On Trainium**: no SGLang → no rollout. BUT the
  `--custom-generate-function-path` hook (`async def generate(args, sample,
  sampling_params) -> Sample`) already exists and is documented — a
  transformers-native `model.generate()` on `neuron` can go there.
- **Fix**: add (or document) a transformers-native generate function on the
  existing hook; no registry/async-server shim needed (unlike verl).

### #2 — Platform abstraction is real but the FSDP backend still hardcodes `torch.cuda`

- **Status**: PARTIAL (source-verified).
- **Class**: package-side (Miles device layer).
- **Observed**: `fsdp_utils/actor.py` (`.to(torch.cuda.current_device())`,
  `.to_empty`, `empty_cache`) and `fsdp_utils/checkpoint.py`
  (`set_rng_state_all`, `synchronize`) — ~10 direct `torch.cuda` sites. The
  `platforms/__init__.py` TODO admits the migration is unfinished.
- **On Trainium**: `torch.cuda.current_device()` raises (no CUDA device).
- **Fix**: migrate these sites to `current_platform` / `torch.accelerator`
  (bounded — ~10 sites, not a file-wide rewrite).

### #3 — No `neuron` platform registered

- **Status**: BLOCKED (source-verified); bounded to fix.
- **Class**: package-side (Miles platform).
- **Observed**: `_select_platform()` accepts only `cuda` and `tpu`.
- **On Trainium**: `MILES_ACCELERATOR=neuron` raises `ValueError`.
- **Fix**: add `NeuronPlatform` (mirror `TpuPlatform`: `dynamic_shapes=False`,
  `nvml_numa_affinity=False`, `device_type="neuron"`) and register it. Verify
  whether `torch.accelerator` exposes the `neuron` PrivateUse1 device; if not,
  fall back to `torch_neuronx.neuron` (same fallback as the verl `PlatformNeuron`).

### #4 — CUDA kernels + Megatron backend are NVIDIA-only

- **Status**: EXCLUDE / FALLBACK (source-verified).
- **Class**: package-side (Miles kernels + Megatron).
- **Observed**: `miles/kernels/` (token-log-softmax, NVFP4 quant, DSA sparse
  attention) are CUDA; the `megatron` backend is NVIDIA-only.
- **On Trainium**: kernels and Megatron do not compile.
- **Fix**: use the `fsdp`/`torchtitan` backends; add eager/CPU fallbacks for the
  kernels the training path actually needs.

### #5 — Docker build bakes CUDA torch + SGLang + Megatron

- **Status**: BLOCKED (source-verified).
- **Class**: package-side (Miles packaging).
- **Observed**: `radixark/miles:latest` ships CUDA torch, SGLang, Megatron;
  `requirements.txt` pins `transformers==5.19.0` (SGLang v0.5.22's pin).
- **On Trainium**: torch must be torch-neuronx (not CUDA torch); SGLang/Megatron
  must be dropped. The `qwen3_5` model needs the transformers line the TRL leg
  pinned (5.18.0); verify 5.19.0 still loads it.
- **Fix**: ship a neuron build (torch-neuronx, no SGLang/Megatron).

---

## Model-level walls (expected, carried from the TRL/verl/SkyRL legs)

Once #1–#3 are fixed and the loop trains on `neuron`, the same torch-neuronx /
neuronx-cc walls hit:

- Static-shape/TDRAM exhaustion from variable RL batch lengths (#23).
- Gated DeltaNet falls back to reference PyTorch without `causal_conv1d` /
  `flash-linear-attention` (#11).
- Dynamic `repeat_interleave` in the vision tower (#1).

---

## Package-side changes we're waiting on (Miles leg)

### miles (framework) — 5

- **#1** — SGLang-only rollout. Fix: add a transformers-native generate function
  on the existing `--custom-generate-function-path` hook (or a `neuron` rollout
  engine).
- **#2** — FSDP backend hardcodes `torch.cuda` (~10 sites). Fix: migrate to
  `current_platform` / `torch.accelerator`.
- **#3** — no `neuron` platform (`MILES_ACCELERATOR` only `cuda`/`tpu`). Fix:
  add `NeuronPlatform`.
- **#4** — Megatron backend + CUDA kernels are NVIDIA-only. Fix: exclude
  Megatron (use fsdp/torchtitan); add eager/CPU kernel fallbacks.
- **#5** — Docker bakes CUDA torch + SGLang. Fix: a neuron build (torch-neuronx,
  no SGLang/Megatron).

### Carried over from the TRL leg (same runtime/compiler stack) — 7

- **torch-neuronx** — #8 unclear OOM, #9 OOM recovery loop, #10 trace-cache
  pin, #11 GDN kernel fallback, #23 TDRAM exhaustion.
- **neuronx-cc** — #1 dynamic `repeat_interleave`, #20 `masked_scatter`
  instruction limit.

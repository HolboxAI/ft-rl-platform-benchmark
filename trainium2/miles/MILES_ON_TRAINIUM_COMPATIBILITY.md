# Miles on Trainium2 (TorchNeuron) — compatibility assessment

Goal: run the same CholecT50 GRPO/RLVR experiment with **Miles** instead of
TRL/verl/SkyRL, and list what library changes a native run needs.

> **Source-verified 2026-10-26 against `radixark/miles` @ `ba60431` (main).**
> No Trainium instance was launched. The one on-instance unknown — whether
> `torch.accelerator` exposes the `neuron` (PrivateUse1) device — is flagged
> below rather than asserted.

## What Miles is

Miles is an open-source RL framework, a fork of ByteDance's `slime`. Rollout
runs on **SGLang**. Training runs on **Megatron-LM**, **PyTorch FSDP**, or
**torchtitan** (FSDP2). It is self-hosted via `train.py` inside a Docker image
(`radixark/miles:latest`). The CholecT50 leg ran FSDP + CPU offload + SGLang on
4×A100.

## Verdict

**Miles is the closest of the three RL frameworks to native Trainium support.**
It already has the two things SkyRL and verl lack, respectively: a device
abstraction with a **non-CUDA precedent** (`TpuPlatform`), and a **documented
rollout escape hatch** (`--custom-generate-function-path`). The port is bounded
— a `NeuronPlatform`, a transformers-native generate function, and ~10
`torch.cuda` call-site migrations in the FSDP backend — not a rewrite.

| Dimension | TRL | verl 0.9.0 | Miles | SkyRL |
|---|---|---|---|---|
| Device layer | native | pluggable `PlatformBase` (CUDA/NPU only) | `Platform` + `torch.accelerator` + **TPU precedent** | none (34 hardcoded `torch.cuda`) |
| Rollout | `model.generate()` | 3 CUDA engines, no escape hatch | SGLang + **`--custom-generate-function-path` hatch** | vLLM only, no hatch |
| Weight sync | n/a | vLLM's own | `cuda_ipc` + `delta` (CUDA-free) | `nccl`/`ipc` + `delta` (CUDA-free) |
| Training backends | — | FSDP | FSDP / torchtitan / Megatron | FSDP / Megatron |

## The good (what Miles already has)

### 1. A device abstraction with a non-CUDA precedent

`miles/platforms/` defines a `Platform` base class and selects one at import via
the `MILES_ACCELERATOR` env var (`cuda` | `tpu`). It calls `torch.accelerator`
for device-agnostic ops (synchronize, empty_cache, memory stats, device index),
and exposes `bind_device_before_process_group` /
`bind_device_after_process_group` / `current_device` / `device_name` /
`dynamic_shapes` / `nvml_numa_affinity`.

`TpuPlatform` is the exact template a `NeuronPlatform` needs:

```python
class TpuPlatform(Platform):
    device_type = "tpu"
    dynamic_shapes = False          # static shapes only — same as neuron
    nvml_numa_affinity = False      # no NVML — same as neuron
    def bind_device_before_process_group(self, local_rank): pass
    def bind_device_after_process_group(self):
        torch.accelerator.set_device_index(0)
```

TPU is the closest analog to Trainium (static shapes, no NVML, device registered
by a `torch_tpu` import). A `NeuronPlatform` is a near-copy.

### 2. A documented rollout escape hatch

The default rollout POSTs to SGLang over HTTP
(`miles/rollout/generate_hub/single_turn.py`), but the CLI has two hooks:

- `--custom-generate-function-path` — a per-sample `async def generate(args,
  sample, sampling_params) -> Sample`. This is exactly where a transformers-native
  `model.generate()` on `neuron` would live.
- `--rollout-function-path` — a whole-rollout replacement
  (`generate_rollout(args, rollout_id, data_source, evaluation=False)`).

The reference CholecT50 leg already uses `--custom-generate-function-path` (to
tweak SGLang's output). The same hook can bypass SGLang entirely.

### 3. A CUDA-free weight-sync protocol

`miles/backends/training_utils/weight_update/protocols/` has both `cuda_ipc` and
`delta`. `delta` syncs weights via checkpoint deltas (disk/S3), no CUDA. Moot
until a rollout exists, but the CUDA-free path is already there.

### 4. Device-agnostic-ish training backends

Three backends: `megatron` (NVIDIA-only), `fsdp` (PyTorch FSDP), `torchtitan`
(FSDP2). TorchNeuron supports FSDP2 + DTensor, so `fsdp` or `torchtitan` is the
neuron path; Megatron is out of scope.

## The blockers (what a native run needs)

### 1. Rollout default is SGLang (CUDA)

SGLang is CUDA/ROCm only. A native run must supply a transformers-native
generate function via the existing `--custom-generate-function-path` hook. This
is the same class of work as verl's `NeuronRollout`, but lower friction: Miles
already documents the hook and its return contract, where verl needed a registry
entry + an async-server shim.

### 2. Platform abstraction is real but incomplete

`Platform` + `torch.accelerator` cover the main device ops, but the FSDP backend
still calls `torch.cuda` directly — ~10 sites in
`fsdp_utils/actor.py` (`.to(torch.cuda.current_device())`, `.to_empty`,
`empty_cache`) and `fsdp_utils/checkpoint.py` (`set_rng_state_all`,
`synchronize`). The `platforms/__init__.py` TODO says the same: "the TPU stack's
remaining call-site checks become platform capabilities, then this goes." The
whole package has 26 files touching `torch.cuda`, but they concentrate in
Megatron (`megatron_utils/`) and CUDA kernels (`kernels/`), not the FSDP path.

### 3. No `neuron` platform registered

`_select_platform()` accepts only `cuda` and `tpu`. Add `NeuronPlatform` +
`MILES_ACCELERATOR=neuron`. The one genuine on-instance unknown: whether
`torch.accelerator` exposes the `neuron` PrivateUse1 device in the torch-neuronx
build. If it does not, the Platform methods that use `torch.accelerator` need a
`torch_neuronx.neuron` fallback (the same fallback our verl `PlatformNeuron`
already does).

### 4. CUDA kernels + Megatron

`miles/kernels/` ships CUDA kernels (token-log-softmax, NVFP4 quant, DSA sparse
attention). The `torch_compile`/eager fallback path in `Platform` covers some
helpers, but the kernels need eager/CPU fallbacks. The Megatron backend is
NVIDIA-only and must be excluded (use `fsdp`/`torchtitan`).

### 5. Dependency build

The Docker image bakes CUDA torch + SGLang + Megatron. A neuron build must swap
torch for torch-neuronx and drop SGLang/Megatron. `requirements.txt` pins
`transformers==5.19.0` (matches SGLang v0.5.22); the CholecT50 model (`qwen3_5`)
needs the transformers line the TRL leg pinned (5.18.0), so a build must verify
5.19.0 still loads `qwen3_5`.

## What a native run needs (package-side)

See `MILES_TRAINIUM_ISSUES_LOG.md` for the numbered list. In short:

1. Add a `NeuronPlatform` (mirror `TpuPlatform`) + `MILES_ACCELERATOR=neuron`.
2. Migrate the FSDP backend's ~10 `torch.cuda` sites to `current_platform` /
   `torch.accelerator`.
3. Add a transformers-native generate function on the existing
   `--custom-generate-function-path` hook.
4. Add eager/CPU fallbacks for `miles/kernels/`; exclude Megatron.
5. Ship a neuron Docker build (torch-neuronx, no SGLang/Megatron).

Plus the same torch-neuronx/neuronx-cc runtime walls the TRL/verl/SkyRL legs hit
(TDRAM, GDN kernels, `repeat_interleave`, OOM ergonomics).

## Ranking the four Trainium legs (closeness to native)

1. **TRL** — works natively (`model.generate()` on neuron).
2. **Miles** — device abstraction + TPU precedent + custom-generate hatch + delta
   sync; bounded port (NeuronPlatform + ~10 FSDP sites + a generate fn).
3. **verl** — device plugin already works, but rollout is async-server-only with
   no escape hatch; needs a registry entry + async shim.
4. **SkyRL** — no device abstraction (34 files) + vLLM-only; a refactor.

## Sources

- <https://github.com/radixark/miles> (inspected @ `ba60431`)
- <https://miles.radixark.com>
- <https://github.com/bytecode-ai/slime> (Miles is a fork of slime)

# TitanRL (torchtitan/rl) on Trainium2 (TorchNeuron) — issues log

Same purpose as the TRL, verl, SkyRL, and Miles issue logs. This one is
**empirically probed on a live `trn2.48xlarge` spot instance** (ap-south-2b,
2026-11-03) plus source-verified against `pytorch/torchtitan` @ `6a2292d`
(cloned on-instance). The probe captured the exact error text from each CUDA
wall, in an isolated venv, without touching the DLAMI torch-neuronx torch.
Full output: `cholec_launch/titanrl_probe.log`. Source assessment:
`TITANRL_ON_TRAINIUM_COMPATIBILITY.md`.

---

## Milestones

- **2026-11-02** — source-verified TitanRL against `69eac07e`. Verdict: trainer +
  weight-sync portable, rollout vLLM-bound + VLM text-only.
- **2026-11-03** — launched `trn2.48xlarge` spot (`i-0b37b5d508e6bbd9c`),
  ran the compatibility probe, pulled the log, terminated the instance.

---

## What is true (empirically confirmed on trn2.48xlarge)

1. The instance exposes 48 neuron cores (12 × 4-core devices), 96 GB per device;
   `torch.device('neuron')` resolves; `torch_neuronx 2.14.3.0.1` loads.
2. `torch.cuda.is_available() = False` and `torch.version.hip = None`.
3. `has_cuda_capability(9, 0) = False` and `(8, 0) = False` — the exact gate
   `generator/vllm/engine.py:283` checks.
4. `pip install torchstore` and `pip install torchmonarch` both fail on PyPI.
5. `pip install vllm` "succeeds" but pulls a full CUDA toolchain (cuda-toolkit
   13.0.3, nvidia-nccl-cu13, nvidia-cudnn-cu13, triton, flashinfer-python).
   `import vllm` works, but `current_platform` is `UnspecifiedPlatform` and
   `DeviceConfig("auto")` raises.
6. `pip install flash-attn` fails (source build, no `torch` in build env).
7. `pip install attn-gym[linear,cudnn]` "succeeds" only because the `cudnn` extra
   is satisfied by NVIDIA cutlass/cuda-toolkit wheels.

---

## Issues (in the order an instance hits them)

### #1 — Weight-sync deps (TorchStore / Monarch) are not installable from PyPI

- **Status**: BLOCKED (probed on-instance).
- **Class**: package-side (install path + PyPI name collision).
- **Observed**:
  ```
  $ pip install torchstore
  ERROR: Cannot install torchstore==0.1.1 and torchstore==0.1.2 because these
  package versions have conflicting dependencies.
  The conflict is caused by:
      torchstore 0.1.2 depends on torchmonarch==0.1.2
      torchstore 0.1.1 depends on torchmonarch==0.1.1
  ERROR: ResolutionImpossible

  $ pip install torchmonarch
  ERROR: Could not find a version that satisfies the requirement torchmonarch
  ERROR: No matching distribution found for torchmonarch
  ```
  The `torchstore` / `torchmonarch` names on PyPI are **unrelated 0.1.x
  packages**. The real Meta TorchStore and Monarch are git-only, installed by
  `scripts/install_rl_env.sh` — which is itself CUDA-bound (see #3/#4/#5).
- **On Trainium**: the one layer that is actually portable (weight sync via
  `direct_rdma=False`) cannot be installed by a normal `pip install`. The
  install path has no neuron branch.
- **Fix**: publish torchstore/torchmonarch to PyPI (or provide a versioned wheel)
  and add a CPU/neuron install path; keep the CUDA RDMA path optional.

### #2 — Rollout `VLLMEngine` cannot infer a device on neuron

- **Status**: BLOCKED (probed on-instance).
- **Class**: package-side (rollout engine) + device-side.
- **Observed**:
  ```
  >>> from torchtitan.tools.utils import has_cuda_capability
  >>> has_cuda_capability(9, 0)  # the gate in generator/vllm/engine.py:283
  False
  ```
  and, against the installed vLLM 0.33.0:
  ```
  >>> from vllm.platforms import current_platform
  >>> current_platform
  <vllm.platforms.interface.UnspecifiedPlatform object ...>
  >>> from vllm.config import DeviceConfig
  >>> DeviceConfig("auto")
  RuntimeError: Failed to infer device type, please set the environment variable
  `VLLM_LOGGING_LEVEL=DEBUG` to turn on verbose logging to help debug the issue.
  ```
  Plus API drift: `EngineArgs` has no `.device` attribute in vLLM 0.33.
- **On Trainium**: TitanRL's only shipped `GenerationEngine` is vLLM; it has no
  `neuron` platform, so `DeviceConfig("auto")` cannot infer a device and the
  rollout never starts.
- **Fix**: implement the `GenerationEngine` ABC (`generator/engine.py`) with
  transformers `model.generate()` on `neuron`, selectable via `Config.engine` —
  the direct analogue of SkyRL's `NeuronInferenceEngine` (PR #2592).

### #3 — `pip install vllm` pulls a CUDA-only toolchain (no neuron wheel)

- **Status**: BLOCKED (probed on-instance).
- **Class**: package-side (vLLM packaging).
- **Observed**: `pip install vllm` exits 0 but installs `cuda-toolkit-13.0.3`,
  `nvidia-nccl-cu13`, `nvidia-cudnn-cu13`, `nvidia-nvshmem-cu13`, `triton`,
  `flashinfer-python`, `torch-2.14.0` (CUDA), etc. There is no neuron vLLM.
  (`vllm-neuron` is a separate beta plugin that torch-neuronx does not include.)
- **On Trainium**: the "install succeeds" is a false pass — the wheels are CUDA
  and the engine still cannot infer a device (#2).
- **Fix**: a neuron vLLM build, or bypass vLLM entirely via #2.

### #4 — `flash-attn` source build fails in the isolated env

- **Status**: BLOCKED (probed on-instance).
- **Class**: package-side (build deps).
- **Observed**:
  ```
  $ pip install flash-attn
  × Getting requirements to build wheel did not run successfully.
  ModuleNotFoundError: No module named 'torch'
  ERROR: Failed to build 'flash-attn' when getting requirements to build wheel
  ```
  flash-attn is a CUDA kernel built from source; its build step needs `torch`
  (not present in the isolated build env), and even with it, it compiles CUDA.
- **On Trainium**: no neuron build of flash-attn exists; it is irrelevant to the
  neuron path but blocks a clean `pip install -r requirements-rl.txt`.
- **Fix**: drop flash-attn from the neuron install; use the eager/neuron path.

### #5 — `attn-gym[linear,cudnn]` resolves to NVIDIA-only wheels

- **Status**: BLOCKED (probed on-instance).
- **Class**: package-side (dependency resolution).
- **Observed**: `pip install attn-gym[linear,cudnn]` exits 0, but the `cudnn`
  extra is satisfied by `nvidia-cutlass-dsl-libs-cu13`, `cuda-toolkit==13.0.3`,
  `nvidia-cudnn-cu13`, `nvidia-nvshmem-cu13`. This is the GDN cuDNN chunk-kernel
  dependency; on neuron the GDN CUDA op (`models/common/attention/gdn.py:87`,
  `device_types="cuda"`) has no counterpart.
- **On Trainium**: the "success" is a CUDA-only tree; the GDN layers still fall
  back to slow reference PyTorch (same wall as TRL/SkyRL `causal_conv1d` /
  `flash-linear-attention`, #11).
- **Fix**: a neuron GDN fallback (eager/reference), and drop the `cudnn` extra
  from the neuron install.

---

## Model-level walls (expected, carried from the TRL/verl/SkyRL/Miles legs)

Once #1–#5 are fixed and the loop trains on `neuron`, the same torch-neuronx /
neuronx-cc walls hit:

- Static-shape/TDRAM exhaustion from variable RL batch lengths (#23).
- Gated DeltaNet falls back to reference PyTorch without `causal_conv1d` /
  `flash-linear-attention` (#11).
- Dynamic `repeat_interleave` in the vision tower (#1).

On top of those, the **CholecT50 VLM modality cannot be expressed in TitanRL at
all**: Qwen3.5 is "text only" for RL (`rl/README.md:36`) and the rollout takes
`prompt_token_ids` (tokens only) — image frames are not representable even on
NVIDIA.

---

## Package-side changes we're waiting on (TitanRL leg)

### torchtitan/rl (framework) — 6

- **#1** — torchstore/torchmonarch not on PyPI; install is git-only + CUDA-bound.
  Fix: publish wheels; add a CPU/neuron install path.
- **#2** — rollout is vLLM-only; `has_cuda_capability(9,0)` gate + no neuron
  platform. Fix: a transformers-native `GenerationEngine` on `neuron` (SkyRL
  #2592 analogue).
- **#3** — vLLM install pulls CUDA nightlies/toolchain. Fix: a neuron vLLM build,
  or bypass vLLM via #2.
- **#4** — flash-attn source build fails (CUDA kernel). Fix: drop it from the
  neuron path.
- **#5** — `attn-gym[cudnn]` resolves to NVIDIA wheels; GDN CUDA op. Fix: a
  neuron GDN fallback.
- **#6** — Qwen3.5 VLM is "text only"; rollout takes tokens, no image modality.
  Fix: a vision branch in the rollout + generator (a feature TitanRL lacks even
  on NVIDIA).

### Carried over from the TRL leg (same runtime/compiler stack) — 7

- **torch-neuronx** — #8 unclear OOM, #9 OOM recovery loop, #10 trace-cache
  pin, #11 GDN kernel fallback, #23 TDRAM exhaustion.
- **neuronx-cc** — #1 dynamic `repeat_interleave`, #20 `masked_scatter`
  instruction limit.

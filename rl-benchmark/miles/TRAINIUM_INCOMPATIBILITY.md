# Why Miles cannot run CholecT50 GRPO on Trainium (torch-neuronx)

_Status as of 2026-10-22. Miles at commit `8e99947` (v0.1.3). Evidence comes from
the Miles source code and an on-hardware probe on a `trn1.2xlarge`._

## Summary

Miles cannot start an RL run on AWS Trainium today. Two independent blockers
each stop it on their own:

1. **No Trainium platform in Miles.** Miles selects its training device from
   `MILES_ACCELERATOR`, which accepts only `cuda` or `tpu`. Any other value
   raises at import time. There is no code path that places the model on
   torch-neuronx's `"neuron"` device.
2. **No rollout engine that runs on Trainium.** Miles generates GRPO rollouts
   with SGLang. Miles has 67 SGLang call sites and no alternative engine.
   SGLang is built for CUDA, and on the Neuron environment it does not even
   install: its dependency resolution fails.

A port would require new code in Miles, not configuration changes (see
[What a port would need](#what-a-port-would-need)). For CholecT50 RL on
Trainium, this benchmark uses stock Hugging Face TRL ([`../hf-trl/`](../hf-trl/))
and a plain-PyTorch GRPO loop ([`../trainium/`](../trainium/)) instead.

## Test setup

| | |
|---|---|
| Instance | `trn1.2xlarge`: 1 Trainium1 chip, 2 NeuronCores, 32 GB HBM (`neuron-ls` confirmed) |
| Image | AWS Neuron DLAMI `torch-neuronx-2.14.0`, Ubuntu 24.04 (newest at test time) |
| Python environment | `/opt/aws_neuronx_venv_torch_neuronx_2_14_0`, Python 3.12.3 |
| PyTorch stack | torch 2.14.0, torch-neuronx 2.14.3, neuronx-cc 2.28; **no `torch_xla`** |
| Miles | `radixark/miles` @ `8e99947`, installed with `pip install --no-deps -e`, plus its light pure-Python dependencies and `transformers==5.19.0`. The torch stack was pinned so pip could not replace it. |
| Run | `miles-trn-20261022-1122-r2`, about 7 minutes, under $0.50 |
| Artifacts | [`logs/miles-trn-20261022-1122-r2/`](logs/miles-trn-20261022-1122-r2/), [`metrics/probe_trainium.json`](metrics/probe_trainium.json) |

Note on torch-neuronx: from Neuron 2.27, torch-neuronx is **TorchNeuron**, a
native PyTorch backend (device `"neuron"`, eager mode + `torch.compile`). It no
longer uses PyTorch/XLA. That is why this image has no `torch_xla`.

## Evidence

### Blocker 1: Miles has no Trainium / Neuron platform

`miles/platforms/__init__.py` chooses the platform once, at import:

```python
def _select_platform() -> Platform:
    device_type = os.environ.get("MILES_ACCELERATOR", "cuda")
    if device_type == "cuda":
        return Platform()
    if device_type == "tpu":
        from miles.platforms.tpu import TpuPlatform   # "TPU v7x through torch_tpu"
        return TpuPlatform()
    raise ValueError(f"MILES_ACCELERATOR={device_type!r}; expected cuda or tpu")
```

On the Trainium host, with `MILES_ACCELERATOR=neuron`:

```
ValueError: MILES_ACCELERATOR='neuron'; expected cuda or tpu
```

Leaving the variable unset doesn't help. It defaults to `cuda`, and the base
`Platform` then calls `torch.cuda.set_device(...)`, which needs an NVIDIA GPU.

The platform check runs when `miles.platforms` is imported. Any module that
imports it fails on Trainium before doing any work. That includes Miles' FSDP
training backend:

| Training backend | Import on Trainium | Note |
|---|---|---|
| `miles.backends.fsdp_utils` | ❌ `ValueError` from the platform check | the backend the CholecT50 GPU run used |
| `miles.backends.torchtitan_utils` | ✅ imports | import only; device placement still goes through `current_platform` |
| `miles.backends.megatron_utils` | ✅ imports | Megatron itself is not installed; Megatron-LM is CUDA-only |

### Blocker 2: the rollout engine (SGLang) does not run on Trainium

Miles' GRPO rollout path is `miles.rollout.sglang_rollout`. The source audit
(production code, tests excluded) counts **67 SGLang call sites in 31 files** and
**0** call sites for any other inference engine. On the Trainium host:

- `import sglang` fails: it is not installable in the Neuron environment.
- `pip install --dry-run sglang` fails during dependency resolution. pip has to
  build a package in SGLang's dependency tree from source, and that build needs
  `apache-tvm-ffi==0.1.0b15`, which is not on PyPI:

  ```
  ERROR: Could not find a version that satisfies the requirement apache-tvm-ffi==0.1.0b15
  ERROR: No matching distribution found for apache-tvm-ffi==0.1.0b15
  × pip subprocess to install build dependencies did not run successfully.
  ```

Even with a successful install, SGLang's kernels target CUDA (Miles' own
requirements list `sgl_kernel`-based builds), and Miles drives it through
CUDA-specific weight sync (NCCL).

### How much of Miles loads at all

The probe imported every Miles module next to torch-neuronx:

| | Count |
|---|---|
| Modules imported | **480 / 645** |
| Failed: `sglang` missing | 51 |
| Failed: `ValueError` (platform check above) | 27 |
| Failed: `megatron` missing | 24 |
| Failed: `sglang_router` missing | 23 |
| Failed: other missing packages (`polars`, `uvicorn`, `tilelang`, `tinker`, …) | 40 |

The failures cluster in exactly the parts a GRPO run needs: rollout (SGLang),
the platform layer, and the Megatron/FSDP training path. The pure-Python
utilities load.

### NVIDIA dependence in the source

`python -m common.porting_audit` over Miles' production code (979 files, 150k
lines; [`metrics/porting_audit.json`](metrics/porting_audit.json)):

| Category | Call sites | Files |
|---|---|---|
| CUDA runtime (`torch.cuda.*`, `.cuda()`, `device="cuda"`) | 141 | 56 |
| NCCL / cupy | 26 | 13 |
| GPU kernels (triton, flash_attn, deep_ep, …) | 62 | 33 |
| NVIDIA stack (megatron, nvidia.*, pynvml) | 323 | 84 |
| SGLang | 67 | 31 |
| JAX | 0 | 0 |
| torch_xla / torch_neuronx / torch_tpu | 1 | 1 (`platforms/tpu.py`) |

## What did *not* fail, and what doesn't count

- **Miles installs and `import miles` works** on the Neuron environment.
  Installing it never touched the torch stack. pip warned only about two
  non-blocking pins (`xxhash`, `zstandard`).
- **The probe's own "accelerator" stage failed, but that is a probe issue, not a
  Miles one.** The probe looked for `torch_xla`, the pre-2.27 torch-neuronx path,
  and this image ships native TorchNeuron instead. The Trainium device itself was
  healthy (`neuron-ls`). The probe and this repo's device layer
  (`common/grpo/device.py`) have since been updated for the native `"neuron"` device.

## What a port would need

Two pieces exist on the AWS side that a port could build on:

- The Neuron DLAMI ships **`torchtitan-neuron` 0.2.2** preinstalled, and Miles
  has a torchtitan training backend.
- AWS publishes **vLLM-on-Neuron** DLAMIs (`pytorch-inference-vllm-*`), a
  Trainium inference engine.

The changes Miles itself would need:

1. **A `NeuronPlatform`** in `miles/platforms/`, analogous to `tpu.py`, that
   places tensors on `"neuron"`. Plus the call-site checks that the TPU path
   still marks `TODO`.
2. **Training on Neuron:** route the torchtitan backend through
   `torchtitan-neuron`, or make the FSDP backend work with TorchNeuron. Both
   need the `torch.cuda` / NCCL call sites in the training path replaced.
3. **A non-SGLang rollout engine:** for example vLLM-on-Neuron, plus a
   weight-update path from the trainer to it that does not use NCCL.
4. **Qwen3.5 on Neuron:** the CholecT50 model's Gated-DeltaNet layers use CUDA
   kernels on GPU; they would need the PyTorch fallback or Neuron kernels.

Until then, Miles' CholecT50 reference result stays GPU-only: val reward
0.319 → 0.651 on 4× A100.

## Reproduce

```bash
bash miles/scripts/run_audit.sh                 # static audit (laptop)
bash miles/trainium/ec2/submit.sh --no-start    # upload probe job
bash miles/trainium/ec2/launch_instance.sh      # trn1.2xlarge, 2 h hard limit
bash miles/trainium/ec2/status.sh --fetch       # results -> miles/logs/<run>/, miles/metrics/
```

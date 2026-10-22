# Miles — CholecT50 GRPO on Trainium and TPU

The Miles CholecT50 GPU run (`ft-rl-platform-benchmark-main 2/miles`, val reward
0.319 → 0.651 on 4× A100) is the reference. This folder asks whether that run
can move to Trainium or TPU, and how far it gets.

## Verdict

Full Trainium write-up with evidence: [`TRAINIUM_INCOMPATIBILITY.md`](TRAINIUM_INCOMPATIBILITY.md).

### From the source (before any hardware run)

| | Trainium | TPU |
|---|---|---|
| Training side | ❌ no platform: `MILES_ACCELERATOR` accepts only `cuda` or `tpu` | ⚠️ **experimental hook**: `miles/platforms/tpu.py`, "TPU v7x through torch_tpu"; undocumented, `IS_TPU` call-site checks still marked TODO |
| Rollout side | ❌ SGLang (CUDA build) | ❌ SGLang (CUDA build); no TPU rollout path in Miles |
| CholecT50 run | not feasible without a new platform + rollout engine | not feasible end-to-end: training could run on TPU, rollouts cannot |
| Experiment type | feasibility probe | feasibility probe (training-side hook) |

Miles at commit `8e99947` (2026-10-22). Docs list only NVIDIA and AMD GPUs under
"Hardware Platforms".

## Porting audit (production code, tests/docs/examples excluded)

`python -m common.porting_audit` over the Miles source, numbers in
[`metrics/porting_audit.json`](metrics/porting_audit.json):

| Category | Call sites | Files |
|---|---|---|
| CUDA runtime (`torch.cuda`, `.cuda()`, `device="cuda"`) | 141 | 56 |
| NCCL / cupy | 26 | 13 |
| GPU kernels (triton, flash_attn, deep_ep, …) | 62 | 33 |
| NVIDIA stack (megatron, nvidia.*, pynvml) | 323 | 84 |
| SGLang (rollout) | 67 | 31 |
| JAX | 0 | 0 |
| torch_xla / torch_neuronx / torch_tpu | 1 | 1 (`platforms/tpu.py`) |

979 Python files, about 150k lines. Compare with SkyRL's JAX path:
[`../skyrl/README.md`](../skyrl/README.md).

## Experiments

| Step | Where | What it shows | Script |
|---|---|---|---|
| A. Static audit | laptop | how much code assumes NVIDIA (done; table above) | `scripts/run_audit.sh` |
| B. TPU probe | TorchTPU VM (`../torchtpu/scripts/setup_env.sh` first) | does the TPU hook bring up a device? which train backends import? does SGLang install? | `tpu/run_probe.sh` |
| C. Trainium probe | trn1 (`../trainium/scripts/setup_env.sh` first) | confirms the missing platform; import surface next to torch-neuronx | `trainium/run_probe.sh` |
| D. CholecT50 run | — | **blocked** on a non-CUDA rollout engine. Not scripted; see config for what a run must match | — |

The probe ([`scripts/probe_miles.py`](scripts/probe_miles.py)) records six
stages to `logs/probe_<chip>.json`: accelerator → `import miles` → platform
device → train backends → rollout (SGLang) → import surface (every module, with
failures grouped by missing package). It was dry-run on a laptop to check the
script itself. On real hardware the first failing stage is the result.

## What would unblock a TPU run

1. **Rollout:** a TPU inference engine Miles can drive. SGLang's JAX/TPU work is
   a separate project (sglang-jax), and its Qwen3.5 vision path was still open
   when last checked. Any engine must also accept weight updates from the trainer.
2. **Training:** finish the `IS_TPU` TODOs. Our TorchTPU leg (`../torchtpu/`) shows
   whether Qwen3.5-4B trains on `torch_tpu` at all.
3. **Hardware:** the hook names TPU v7x (Ironwood). Check availability for this
   project before booking v6e and assuming parity.

## Contents

| Path | What |
|---|---|
| `config/MILES_CHOLECT50_CONFIG.md` | reference GPU settings any port must match, and findings log |
| `reference_gpu/` | the GPU run's scripts and results, verbatim |
| `scripts/probe_miles.py`, `scripts/run_audit.sh` | probe + audit |
| `tpu/`, `trainium/` | per-chip probe launchers |
| `logs/`, `metrics/` | probe JSON, audit JSON |

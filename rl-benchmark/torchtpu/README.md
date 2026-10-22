# TorchTPU (Google Cloud TPU): CholecT50 GRPO with native PyTorch on TPU

This folder is the **TPU leg** of the accelerator benchmark. It runs the shared
GRPO trainer ([`../common/grpo/`](../common/grpo/)) on Cloud TPU through
**TorchTPU** (`torch_tpu`), Google's PyTorch-native TPU backend (device `"tpu"`,
eager mode + `torch.compile`). The task is CholecT50 only.

The same TPU VM also runs the trainer on **PyTorch/XLA** (`torch_xla`, the
previous way to use TPUs from PyTorch). That answers "is TorchTPU better than
torch_xla for this workload?" on identical hardware.

## Status

| Phase | What | Path | Status | Results |
|---|---|---|---|---|
| 0 | Smoke test (SFT checkpoint + one val frame at the run's shapes) | TorchTPU | ⏳ | `logs/smoke_torchtpu.json` |
| 0 | Smoke test | PyTorch/XLA | ⏳ | `logs/smoke_torchxla.json` |
| 1 | CholecT50 GRPO, LoRA 16, 16 steps | TorchTPU eager | ⏳ | `metrics/cholect50_rl_torchtpu_final_results.json` |
| 1 | same | TorchTPU + `torch.compile` | ⏳ | `metrics/cholect50_rl_torchtpu-compile_final_results.json` |
| 1 | same | PyTorch/XLA | ⏳ | `metrics/cholect50_rl_torchxla_final_results.json` |

## Run it

```bash
# 1. Create the TPU VM (from your laptop)
PROJECT=... ZONE=... ACCEL=v6e-1 bash torchtpu/scripts/launch_tpu_vm.sh

# 2. Copy this repo to the VM, then on the VM:
bash torchtpu/scripts/setup_env.sh                                   # both venvs
bash common/fetch_cholect50.sh                                       # needs AWS read access to the bucket
bash torchtpu/scripts/run_smoke.sh                                   # TorchTPU
TPU_PATH=torchxla bash torchtpu/scripts/run_smoke.sh                 # PyTorch/XLA

bash torchtpu/scripts/run_cholect50_rl.sh --num-steps 1              # one step
bash torchtpu/scripts/run_cholect50_rl.sh                            # full, eager
COMPILE=1 bash torchtpu/scripts/run_cholect50_rl.sh                  # full, torch.compile
TPU_PATH=torchxla bash torchtpu/scripts/run_cholect50_rl.sh          # full, baseline
```

Delete the VM when done. TPU VMs bill while they exist, even when idle.

## What to watch for

- **Getting the data.** The SFT checkpoint and parquet live in AWS S3. Either
  give the VM read-only AWS credentials for `fetch_cholect50.sh`, or copy them
  to a GCS bucket first.
- **Install line.** TorchTPU is new. Use the exact install command and runtime
  version from the Cloud TPU "PyTorch on TPU" docs, and record it in
  `config/TORCHTPU_CONFIG.md`. Set `TORCH_TPU_PIP` to override the default.
- **Eager vs compile.** The static sampler runs 256 full forwards per rollout
  batch, including the vision tower. Eager dispatch overhead shows up directly
  in `gen_tok_per_s`, and `COMPILE=1` should close that gap if `torch.compile`
  handles Qwen3.5 on TorchTPU. Whether it does is a finding.
- **Synchronisation.** The trainer looks up `torch.tpu.synchronize` for timing.
  If the installed TorchTPU names it differently, step timings will be skewed;
  check `common/grpo/device.py:TorchTpuBackend` against the docs.

## Contents

| Path | What it holds |
|---|---|
| `config/TORCHTPU_CONFIG.md` | hardware, versions, hyperparameters, findings log |
| `scripts/launch_tpu_vm.sh` | `gcloud` TPU VM create |
| `scripts/setup_env.sh` | two venvs (TorchTPU, PyTorch/XLA) + `logs/environment.txt` |
| `scripts/tpu_env.sh` | selects the venv and backend (`TPU_PATH=torchtpu|torchxla`) |
| `scripts/run_smoke.sh` | Phase 0 |
| `scripts/run_cholect50_rl.sh` | Phase 1 |
| `logs/`, `metrics/` | as in the Trainium leg |

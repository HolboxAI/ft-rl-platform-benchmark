# Trainium (AWS): CholecT50 GRPO with torch-neuronx

This folder is the **torch-neuronx leg** of the accelerator benchmark. It runs
the shared GRPO trainer ([`../common/grpo/`](../common/grpo/)) on AWS Trainium
through **torch-neuronx** (PyTorch/XLA + the Neuron compiler), with no CUDA in
the stack. The task is CholecT50 only.

## Status

| Phase | What | Status | Results |
|---|---|---|---|
| 0 | Smoke test: SFT checkpoint + one val frame at the run's shapes (forward, LoRA step, sampling) | ⏳ not started | `logs/smoke.json` |
| 1 | CholecT50 GRPO: Qwen3.5-4B VLM, LoRA 16, 16 steps | ⏳ blocked on Phase 0 | `metrics/cholect50_rl_final_results.json` |
| — | Full-model CholecT50 GRPO (sharded across NeuronCores) | 🔲 out of scope for this trainer | — |

## What it measures

The RL frameworks used so far (Miles, verl, SkyRL, NeMo-RL) all generate
rollouts with CUDA engines (SGLang / vLLM), so none of them run on Trainium
as-is. This leg measures what CholecT50 GRPO costs on Trainium when the loop
is plain PyTorch:

- **Does it work?** Which stage fails first, if any. The smoke test records it.
  Qwen3.5's Gated-DeltaNet layers must use the pure-PyTorch fallback here.
- **Compile cost.** XLA compiles every new tensor shape. The trainer keeps
  shapes static (2,048-token prompts, 256-token responses, fixed micro-batches).
  `first_step_s` vs `steady_step_s` shows the overhead.
- **Throughput and cost.** Step time, generated tokens/s and $ per run, compared
  with the GPU legs.
- **Quality.** Val reward before and after. Compare it with the Tinker LoRA run
  (0.364 → 0.419) and the SkyRL GPU control run's `tinker` preset.

## Run it

```bash
# 1. Launch (from your laptop)
INSTANCE_TYPE=trn1.2xlarge KEY_NAME=... SUBNET_ID=... SG_ID=... \
  bash trainium/scripts/launch_instance.sh

# 2. Copy this repo to the instance, then on the instance:
bash trainium/scripts/setup_env.sh
bash common/fetch_cholect50.sh                         # SFT checkpoint + parquet from S3
bash trainium/scripts/run_smoke.sh                     # Phase 0
bash trainium/scripts/run_cholect50_rl.sh --num-steps 1  # Phase 1, one step
bash trainium/scripts/run_cholect50_rl.sh              # Phase 1, full (16 steps)
```

Monitor utilisation with `neuron-top` in a second shell. Compiled graphs are
cached in `$NEURON_COMPILE_CACHE_URL`, so a rerun skips most compile time.
Record whether a run was cold or warm in `config/TRAINIUM_CONFIG.md`.

## Contents

| Path | What it holds |
|---|---|
| `config/TRAINIUM_CONFIG.md` | hardware, SDK versions, every hyperparameter, findings log |
| `scripts/launch_instance.sh` | EC2 launch on the Neuron DLAMI |
| `scripts/neuron_env.sh` | venv + compiler/runtime env vars (sourced by every script) |
| `scripts/setup_env.sh` | one-time install; records `logs/environment.txt` |
| `scripts/run_smoke.sh` | Phase 0 |
| `scripts/run_cholect50_rl.sh` | Phase 1 |
| `logs/` | console logs and per-run folders, verbatim |
| `metrics/` | `cholect50_rl_final_results.json` (schema: `../analysis/schema.md`) + per-step `.jsonl` |

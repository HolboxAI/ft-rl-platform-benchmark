# SkyRL — CholecT50 GRPO on TPU and Trainium

The SkyRL CholecT50 GPU run (`ft-rl-platform-benchmark-main 2/skyrl`, FSDP +
vLLM, reward collapsed 0.329 → 0.0) is the reference. SkyRL's GPU trainer is
CUDA-bound, but SkyRL also ships a **JAX backend behind its Tinker-compatible
API server**. JAX runs on TPU natively and on Trainium through AWS's beta JAX
NeuronX plugin. So the accelerator experiment here is:

> the **same CholecT50 RL client** (a port of the Tinker leg's script) talks to a
> self-hosted SkyRL Tinker server, whose backend is FSDP on GPU (control), JAX
> on TPU, or JAX on Trainium.

## Verdict so far (from the source, before any hardware run)

| | GPU control (FSDP) | TPU (JAX) | Trainium (JAX on jax-neuronx) |
|---|---|---|---|
| Server runs on the chip | ✅ documented | ✅ documented (`--extra tpu`) | ⚠️ untested; jax-neuronx is beta, and SkyRL needs jax ≥ 0.8 |
| Qwen3.5-4B (Gated DeltaNet) | ✅ | ✅ text model implemented (`skyrl/tx/models/qwen3_5.py`, chunked GDN in JAX) | ⚠️ same model code; Neuron compile unknown |
| LoRA + `importance_sampling` loss | ✅ | ✅ (JAX supports `cross_entropy`, `importance_sampling`, `ppo`) | ⚠️ |
| **Image input (CholecT50 needs it)** | ✅ (vLLM render path) | ❌ **JAX Qwen3.5 is text-only**; no `pixel_values` handling in the JAX backend | ❌ same |
| KL penalty | ❌ not via Tinker API (`kl_penalty_coef > 0` unsupported) | ❌ | ❌ |
| CholecT50 run | ✅ runnable (control) | **blocked at images**; probe stages 1–4 runnable | **blocked at images**; probe runnable |

SkyRL at commit `b304652` (2026-10-21).

## Porting audit

| Scope | Files / lines | NVIDIA-specific call sites | JAX call sites |
|---|---|---|---|
| All SkyRL production code | 378 / 110k | 388 (CUDA 172, NCCL 56, kernels 20, NVIDIA stack 140) + 131 vLLM | 62 |
| `skyrl/tx` (JAX models + layers) | 25 / 5.0k | **0** | 53 |

The JAX path has no NVIDIA dependencies at all; Miles has no JAX path. That is
why SkyRL gets a real experiment here and Miles gets a probe. Numbers are in
[`metrics/porting_audit*.json`](metrics/).

## Experiments

| Step | Where | What | Script |
|---|---|---|---|
| 0. Setup | each host | clone SkyRL @ `b304652`, client venv (tinker + tinker-cookbook) | `scripts/setup_skyrl.sh` |
| 1. Stack probe | TPU / Trainium / GPU | server on `Qwen/Qwen3-0.6B`; probe stages connect → CE step → IS-loss step → sample → image | `<chip>/start_server.sh` + `<chip>/run_probe.sh` |
| 2. Model probe | same | server on the CholecT50 SFT checkpoint (Qwen3.5-4B); same probe. Expected on TPU / Trainium: pass 1–4, **fail at `image_input`** | same, `MODEL=~/models/cholec-sft-qwen3.5-4b` |
| 3a. GPU control, `tinker` preset | g5.12xlarge | CholecT50 RL through the Tinker API on FSDP, Tinker-leg recipe | `gpu_control/start_server.sh` + `PRESET=tinker gpu_control/run_cholect50_rl.sh` |
| 3b. GPU control, `skyrl-grpo` preset | g5.12xlarge (restart server first) | same, as close to the existing SkyRL run as possible (full model, clipped GRPO loss, std-normalised advantages, 8 updates/step, temp 0.7, KL 0) | `PRESET=skyrl-grpo gpu_control/run_cholect50_rl.sh` |
| 4. Accelerator run | TPU (then Trainium) | same client; runs once step 2 passes `image_input` | `tpu/run_cholect50_rl.sh` |

### Running on TPU

```bash
# TPU VM from ../torchtpu/scripts/launch_tpu_vm.sh (v6e-1 for 0.6B; v6e-4 for Qwen3.5-4B with FSDP 4)
bash skyrl/scripts/setup_skyrl.sh
MODEL=Qwen/Qwen3-0.6B bash skyrl/tpu/start_server.sh        # shell 1, wait for "Uvicorn running"
MODEL=Qwen/Qwen3-0.6B bash skyrl/tpu/run_probe.sh           # shell 2
# then restart the server with MODEL=$HOME/models/cholec-sft-qwen3.5-4b and probe again
```

## The CholecT50 client

[`scripts/cholect50_rl_client.py`](scripts/cholect50_rl_client.py) is the Tinker
leg's `tinker_rl.py` with `base_url` added. The training loop is
tinker_cookbook's RL trainer, unchanged; SkyRL documents that cookbook recipes
run unmodified with `base_url`. It differs from the Tinker leg in three
deliberate ways:

- It starts from the merged SFT checkpoint served as the base model, with a fresh
  LoRA 16. That's the same weights as verl / Miles / SkyRL-GPU.
- It uses the shared 256 / 32 verl parquet.
- It runs a greedy val eval before and after.

Its `--preset` picks the recipe:

- **`tinker`** (default) is the Tinker leg's recipe: LoRA 16, `importance_sampling` loss,
  mean-centred advantages, one update per step, temperature 1.0.
- **`skyrl-grpo`** is the existing SkyRL GPU run as far as the Tinker API allows: full model,
  `ppo` loss (mapped by SkyRL's FSDP backend to its `regular` GRPO policy loss, ε 0.2),
  advantages divided by the group std (patched into the cookbook client), 8 updates per step,
  temperature 0.7, constant-reward groups kept. KL stays 0, because SkyRL's Tinker backend
  forces it off.

Each setting can also be set on its own (`--loss`, `--norm-adv-by-std`,
`--num-substeps`, `--lora-rank`, `--temperature`), which allows one-variable
ablations. For example, `--preset skyrl-grpo --num-substeps 1` tests whether
the 8 updates per step drive the collapse. Full parity table:
[`config/SKYRL_CHOLECT50_CONFIG.md`](config/SKYRL_CHOLECT50_CONFIG.md).

## Contents

| Path | What |
|---|---|
| `config/SKYRL_CHOLECT50_CONFIG.md` | settings, parity table, probe results, findings log |
| `reference_gpu/` | the GPU run's scripts, config and results, verbatim |
| `scripts/` | client, probe, setup |
| `tpu/`, `trainium/`, `gpu_control/` | per-platform server + run scripts |
| `logs/`, `metrics/` | server logs, probe JSON, results, audits |

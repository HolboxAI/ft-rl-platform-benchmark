# rl-benchmark: CholecT50 GRPO on AWS Trainium (torch-neuronx)

Self-contained folder. Nothing outside `rl-benchmark/` is changed or imported.
It asks whether the CholecT50 surgical action-triplet GRPO run (the verl /
Miles / SkyRL / NeMo-RL / Tinker legs elsewhere in this repo) can run on **AWS
Trainium through torch-neuronx**, and records exactly where each stack stops.

**CholecT50 only:** the same verl parquet (256 train / 32 val frames), the same
merged SFT checkpoint (`cholec-sft-qwen3.5-4b`, Qwen3.5-4B VLM) and the same
triplet-F1 reward as the GPU legs.

## Contents

| Folder | Experiment | Status (2026-10-22) |
|---|---|---|
| [`miles/`](miles/) | Can Miles run on Trainium? Static porting audit + on-hardware probe | ❌ **Not compatible.** No Neuron platform (`MILES_ACCELERATOR` accepts only `cuda`/`tpu`), and SGLang does not install on the Neuron venv. Evidence: [`miles/TRAINIUM_INCOMPATIBILITY.md`](miles/TRAINIUM_INCOMPATIBILITY.md) |
| [`hf-trl/`](hf-trl/) | Stock Hugging Face TRL `GRPOTrainer` on native TorchNeuron | 🟡 TorchNeuron, accelerate (picks `"neuron"`), TRL / transformers / peft and the Qwen3.5 vision tower all work. Blocked in `generate` prefill (int64 `input_ids` corrupted). Log: [`hf-trl/config/TRL_CHOLECT50_CONFIG.md`](hf-trl/config/TRL_CHOLECT50_CONFIG.md) |
| [`common/`](common/) + [`trainium/`](trainium/) | Our own plain-PyTorch GRPO loop (no RL framework) on native TorchNeuron | 🟢 First Trainium run in progress (smoke test + 1 GRPO step) |

Supporting files: [`analysis/`](analysis/) (metrics schema, comparison script),
[`data/`](data/) (CholecT50 data notes; the parquet is fetched from S3, not
committed), [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) (also covers a TPU leg
that is not part of this folder).

## Environment (all three experiments)

| | |
|---|---|
| Instance | `trn1.2xlarge` (1 Trainium1 chip, 2 NeuronCores, 32 GB HBM), us-east-1 |
| AMI | Neuron DLAMI `torch-neuronx-2.14.0`, Ubuntu 24.04 |
| Stack | torch 2.14.0, torch-neuronx 2.14.3 (**native TorchNeuron**, device `"neuron"`, no `torch_xla`), neuronx-cc 2.28 |
| Launch | S3-driven EC2 runner per experiment (`*/ec2/` or `*/trainium/ec2/`): `submit.sh` → `launch_instance.sh` → `status.sh --fetch`. No SSH; hard time limit; results synced to S3 |

## Findings so far

- **Native TorchNeuron works on Trn1.** The Neuron docs list Trn2 / Trn3, but
  `torch.neuron.is_available()` is true and tensor ops run on `trn1.2xlarge`.
- **accelerate ≥ 1.15 selects `torch.device("neuron")` by itself.** TRL needs
  no device code.
- **Qwen3.5 vision on TorchNeuron** needs its data-dependent index helpers
  (`repeat_interleave` with tensor repeats, mRoPE indices) run on CPU:
  [`common/neuron_compat.py`](common/neuron_compat.py). After that the vision
  tower compiles (~12 min cold) and runs.
- **Op correctness on neuron vs CPU (20 cases):** 18 match. `int64.sum()` and
  `int64.cumsum()` are wrong for large values (float32 accumulation; garbage
  cumsum). Mask and count ops used by RL are correct at realistic sizes.
- **Miles:** 552 NVIDIA-specific + 67 SGLang call sites in production code,
  0 JAX. Not portable without new platform and rollout code.

## Quick start

```bash
cd rl-benchmark
# CPU correctness tests for the GRPO loop (any small Qwen3 text checkpoint)
pip install torch -r requirements-common.txt
python -m common.tests.test_rollout --model Qwen/Qwen3-0.6B
python -m common.tests.test_grpo_direction Qwen/Qwen3-0.6B

# On AWS (profile stanford-gpu): our GRPO loop on Trainium
bash trainium/ec2/submit.sh --no-start && bash trainium/ec2/launch_instance.sh
bash trainium/ec2/status.sh --fetch
```

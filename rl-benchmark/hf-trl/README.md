# Hugging Face TRL on Trainium: CholecT50 GRPO with native TorchNeuron

This folder tests **stock TRL (`GRPOTrainer`)** on AWS Trainium through
**torch-neuronx**, on the CholecT50 task only. (It is named `hf-trl/`, not
`trl/`, so it can't shadow the `trl` package when scripts run from the repo root.)

## Why TRL can work here when Miles / SkyRL cannot

| Piece | What makes it Trainium-capable | Checked |
|---|---|---|
| torch-neuronx 2.14 (DLAMI `torch-neuronx-2.14.0`) | **Native TorchNeuron backend**: device `"neuron"` (PyTorch PrivateUse1), eager + `torch.compile`, no `torch_xla` | on a trn1.2xlarge, 2026-10-22: venv has torch 2.14.0 + torch-neuronx 2.14.3, **no torch_xla** |
| accelerate ≥ 1.15 | `is_neuron_available()` → `PartialState().device == torch.device("neuron")`; `MULTI_NEURON` for several cores | source of accelerate 1.15.0 |
| TRL 1.16 `GRPOTrainer` | Rollouts with `transformers.generate` (no vLLM needed); images via an `image` column | source of trl 1.16.0 |
| optimum-neuron 0.4.3 | ❌ not used: no GRPO trainer, pins XLA-based torch-neuronx 2.8 and transformers 4.57 (pre-Qwen3.5) | wheel metadata |

So the script has **no Neuron-specific code**. Whether stock TRL works
unchanged on the native backend is the result.

Open risk: the Neuron docs list TorchNeuron for **Trn2 / Trn3**. This account's
Trainium quota only fits Trn1 (`trn1.2xlarge`), so the first stage records
whether the native backend runs on Trn1 at all.

## Experiment

| | |
|---|---|
| Model | merged SFT checkpoint `cholec-sft-qwen3.5-4b` (Qwen3.5-4B VLM), bf16, **LoRA 16** |
| Data | CholecT50 verl parquet: 256 train / 32 val frames; prompts > 2048 tokens dropped |
| GRPO | 16 frames × 8 samples = 128 completions per optimizer step, **16 steps** (1 epoch) |
| Sampling | temperature 0.7, top-p 1, no top-k, 256 tokens, thinking disabled |
| Loss | `dapo` (token-level over active tokens), clip 0.2 / 0.28, KL β 0.04, group-std advantages |
| Optimizer | AdamW, lr 1e-5 constant, betas (0.9, 0.98), wd 0.1 |
| Reward | 0.4 × triplet F1 + 0.6 × mean(instrument, verb, target F1) |
| Eval | greedy, 32 val frames, before and after |
| Compare to | Tinker LoRA 0.364 → 0.419; torch-neuronx leg (`trainium/`); SkyRL GPU control `tinker` preset |

## Stages (`trainium/ec2/job.sh`)

| Stage | What | Result file |
|---|---|---|
| env | Neuron venv, versions, `neuron-ls` | `environment.txt` |
| install | trl 1.16.0, accelerate 1.15.0, transformers 5.19.0, peft (torch stack frozen) | `install.txt` |
| data | checkpoint + parquet from S3, hashes | `data_manifest.txt` |
| check | `scripts/check_env.py`: neuron tensor op → accelerate device → imports → model load + 16-token generate | `check_env.json` |
| smoke | 1 GRPO step, generation in chunks of 16 completions | `smoke/final_results.json` |
| full | 16 steps + val eval before/after (add `full` to `STAGES`, raise `HARD_LIMIT_HOURS`) | `full/final_results.json` |

```bash
bash hf-trl/trainium/ec2/submit.sh --no-start      # upload code + job
bash hf-trl/trainium/ec2/launch_instance.sh        # trn1.2xlarge, 3 h hard limit
bash hf-trl/trainium/ec2/status.sh [--fetch]
```

## Contents

| Path | What |
|---|---|
| `scripts/train_grpo_trl.py` | TRL GRPO training + eval + `final_results.json` |
| `scripts/check_env.py` | staged environment check |
| `config/TRL_CHOLECT50_CONFIG.md` | settings, versions, findings log |
| `trainium/ec2/` | S3-driven EC2 runner (launch / submit / status / job) |
| `logs/`, `metrics/` | fetched run outputs |

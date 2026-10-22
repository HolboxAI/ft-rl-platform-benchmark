# DS-1000 — LoRA + GRPO on Trainium (TRL + torch-neuronx)

Direct RL (RLVR) leg for the platform benchmark: fine-tune a small **Qwen2.5-Coder**
model to solve **DS-1000** data-science programming problems, scoring reward as
pass/fail on each problem's hidden test harness.

| | |
|---|---|
| Dataset | [DS-1000](https://github.com/xlang-ai/DS-1000) — 1,000 problems, 7 libraries |
| Task | generate Python that passes hidden `test_execution` / `test_string` tests |
| Model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (LoRA, r=16) |
| Algorithm | GRPO (RLVR) via TRL `GRPOTrainer` |
| Reward | 1.0 pass / 0.0 fail / -0.5 empty, sandboxed subprocess |
| Hardware | AWS Trainium (trn1.2xlarge+, DLAMI) |

## Run

```bash
# smoke (Pandas subset, 1 epoch, G=4)
jupyter notebook ds1000_lora_grpo.ipynb

# full (all 1,000 problems, 3 epochs, G=8)
FULL_RUN=1 jupyter notebook ds1000_lora_grpo.ipynb
```

## Why TRL, not SkyRL / verl / TitanRL

Those frameworks hardwire a **vLLM** rollout (CUDA-only). TRL rolls out with Transformers
`model.generate`, which `torch-neuronx` supports — that is the only RL path on Trainium today.

## Key config (notebook §4)

| Parameter | Smoke | Full |
|---|---|---|
| Model | Qwen2.5-Coder-1.5B-Instruct | same |
| Library | Pandas (160 problems) | all (1,000) |
| Epochs | 1 | 3 |
| num_generations (G) | 4 | 8 |
| lr (LoRA) | 1e-4 | 1e-4 |
| beta (KL) | 0.04 | 0.04 |
| temperature | 0.7 | 0.7 |
| cache_implementation | static | static |
| pad_to_multiple_of | 128 | 128 |

## Metric

pass@1 on a held-out 20% split, greedy decode, per library. Report before-vs-after pass@1
plus the GRPO training reward curve.

## Safety

`check_correctness` is a *timeout + process* guard, **not** a security sandbox. DS-1000 code
legitimately touches the filesystem (Matplotlib/TF/sklearn), so run on an isolated Trainium
instance and never on a shared machine.

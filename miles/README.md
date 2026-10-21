# Miles — RL leg

Fourth platform in the fine-tuning + RL benchmark. Miles is an open-source RL
framework (fork of `slime`): SGLang for rollout, Megatron-LM or PyTorch FSDP for
training. It is self-hosted, like Prime Intellect.

This leg covers the two RL experiments (GRPO). SFT stays on the other three
platforms.

## Files

- [`config/MILES_RL_CONFIG.md`](config/MILES_RL_CONFIG.md) — portable RL config,
  reward contract, data format, and findings.
- [`DECISIONS_LOG.md`](DECISIONS_LOG.md) — decisions and reasoning.
- [`PROMPTS_AND_INTENTS.md`](PROMPTS_AND_INTENTS.md) — the brief and what the
  agent did with it.
- [`scripts/reward_orena.py`](scripts/reward_orena.py) — ORena reward, port of
  the Fireworks evaluator. Offline smoke test included.
- [`scripts/reward_goemotions.py`](scripts/reward_goemotions.py) — GoEmotions
  reward (Jaccard). Offline smoke test included.
- [`scripts/prepare_data.py`](scripts/prepare_data.py) — builds Miles JSONL from
  the canonical `data/` files.
- [`scripts/run_goemotions_rl.sh`](scripts/run_goemotions_rl.sh) — launch,
  FSDP backend, de-risk leg.
- [`scripts/run_orena_rl.sh`](scripts/run_orena_rl.sh) — launch, Megatron
  backend + VLM generate function, follow-on.
- [`scripts/generate_orena.py`](scripts/generate_orena.py) — VLM generate
  function (template).
- [`scripts/log_rollout_orena.py`](scripts/log_rollout_orena.py) — rollout-log
  hook that reports `exact_match` alongside `reward_value` (correctness).
- [`scripts/extract_orena_rl_metrics.py`](scripts/extract_orena_rl_metrics.py) —
  parses a run log into per-step curves and writes `metrics/orena_rl_final_results.json`.
- [`data/goemotions_rl_miles.jsonl`](data/goemotions_rl_miles.jsonl) — 5,000
  rows, committed, runnable now.
- [`data/orena_rl_miles_sample.jsonl`](data/orena_rl_miles_sample.jsonl) — 2
  rows, smoke only. Full ORena data is regenerated before the run.
- [`metrics/run_manifest.json`](metrics/run_manifest.json) — status.
- [`metrics/orena_rl_final_results.json`](metrics/orena_rl_final_results.json) —
  ORena RL result (curves + summary), comparable to Fireworks' `orena_rl` block.
- [`metrics/cholect50_rl_final_results.json`](metrics/cholect50_rl_final_results.json) —
  CholecT50 RL result (curves + summary), a strict reproduction of the verl
  `cholec-t50-grpo` reference (target val greedy mean@1 = 0.715).
- [`scripts/reward_cholect50.py`](scripts/reward_cholect50.py) — CholecT50
  surgical-triplet reward (F1-blended), verbatim port of the verl reward.
- [`scripts/generate_cholect50.py`](scripts/generate_cholect50.py) — VLM generate
  function (Qwen3.5 template).
- [`scripts/log_rollout_cholect50.py`](scripts/log_rollout_cholect50.py) —
  rollout-log hook.
- [`scripts/prepare_cholect50_data.py`](scripts/prepare_cholect50_data.py) —
  verl parquet → Miles JSONL.
- [`scripts/run_cholect50_rl.sh`](scripts/run_cholect50_rl.sh) — launch, FSDP
  backend + CPU offload.
- [`scripts/extract_cholect50_rl_metrics.py`](scripts/extract_cholect50_rl_metrics.py) —
  parses a run log into `metrics/cholect50_rl_final_results.json`.
- [`logs/run_orena.log`](logs/run_orena.log) — ORena RL console log (spot run).
- [`logs/run_cholect50.log`](logs/run_cholect50.log) — CholecT50 RL console log
  (spot run). `run_cholect50_broken.log` is the first, reward-zero run that
  documented the EOS-token bug before the fix.

## Status

Three RL legs are complete. GoEmotions RL ran on a g5.2xlarge (FSDP); ORena VLM
RL ran on a p4d.24xlarge spot node (Megatron); CholecT50 GRPO ran on a
p4d.24xlarge spot node (FSDP, 4×A100). Results are in `metrics/` and the
comparison charts are in `analysis/plots/`. The CholecT50 leg (val 0.651 vs verl
0.715) has three documented divergences from the verl reference — see the
`cholect50_rl` findings in `metrics/run_manifest.json`.

## Run it

1. Start a GPU node on the `riley` AWS profile.
2. Pull the Miles image: `docker pull radixark/miles:latest`.
3. Copy `scripts/` and `data/goemotions_rl_miles.jsonl` to the node.
4. Inside the container, run `BENCH_DIR=/root/miles_bench bash run_goemotions_rl.sh`.

For the ORena VLM leg, `run_orena_rl.sh` already bakes in the three A100
gotchas (uninstall flash-attn-4, `--num-epoch 1` not `--num-rollout`, and the
exact_match log hook) — see `metrics/run_manifest.json` findings.

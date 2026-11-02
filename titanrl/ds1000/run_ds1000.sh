#!/usr/bin/env bash
# TitanRL GRPO on DS-1000 (Qwen3-0.6B, 4x A10G = 1 trainer + 1 generator).
set -euo pipefail

cd ~/torchtitan
source ~/titan-rl/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 100 training steps, 8 prompts/step, 4 samples/prompt (~800 DS-1000 problems).
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.ds1000 \
  --config rl_grpo_qwen3_0_6b_ds1000 \
  --output-dir outputs/ds1000 2>&1 | tee ~/ds1000_train.log

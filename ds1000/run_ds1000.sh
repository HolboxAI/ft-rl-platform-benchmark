#!/usr/bin/env bash
set -xeuo pipefail

export VERL_PLATFORM=tpu
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export TPU_ACCELERATOR_TYPE=v6e
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export PYTHONPATH=/verl:/verl-hardware-plugin:/torchtitan

# 8-chip v6e single host: 4 trainer + 4 rollout
export NNODES_TRAINER=1 N_CHIPS_TRAINER=4
export NNODES_ROLLOUT=1 N_CHIPS_ROLLOUT=4

export MODEL_NAME=Qwen3.5-4B
export MODEL_PATH=/models/Qwen3.5-4B-text
export TRAIN_FILE=/workspace/parquet/train.parquet
export TEST_FILE=/workspace/parquet/val.parquet
export TOTAL_TRAINING_STEPS=10
export VAL_BEFORE_TRAIN=False
export VERL_TPU_TIMELINE=0
export OPTIM_IMPL=fused
export TPU_ROLLOUT_CHIP_OFFSET=$((NNODES_TRAINER * N_CHIPS_TRAINER))
export GPU_MEMORY_UTILIZATION=0.4
export MAX_PROMPT_LEN=1024
export MAX_RESPONSE_LEN=1024

ray stop --force >/dev/null 2>&1 || true
ray start --head --resources='{"TPU": 8}' --disable-usage-stats >/tmp/ray_start.log 2>&1

bash /workspace/run_tpu.sh \
  +ray_kwargs.ray_init.runtime_env.env_vars.TPU_ROLLOUT_CHIP_OFFSET="'${TPU_ROLLOUT_CHIP_OFFSET}'" \
  reward.custom_reward_function.path=/workspace/ds1000_reward.py \
  reward.custom_reward_function.name=compute_score \
  ++data.apply_chat_template_kwargs.enable_thinking=False

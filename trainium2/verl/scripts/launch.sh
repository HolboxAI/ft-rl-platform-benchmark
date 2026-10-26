#!/usr/bin/env bash
# FIRST-ATTEMPT verl RL launch on Trainium2. NOT expected to succeed as-is.
#
# This exists to capture the first genuine failures after the gate passes.
# The known blocker is issue #1 in VERL_TRAINIUM_ISSUES_LOG.md: verl 0.9.1
# removed the sync/transformers-native rollout, so `rollout.name=naive` is a
# guess to be resolved on-instance (older verl, a custom async adapter, or a
# registry monkeypatch). Run gate_test.py first; do not run this blind.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"

export VERL_PLATFORM=neuron
export NEURON_RT_VISIBLE_CORES=0
export VLLM_USE_MODELSCOPE=False

# Data is built by verl/scripts/cholec_data.py into ~/cholec/parquet/.
DATA_DIR="$HOME/cholec/parquet"

cd "$REPO/verl"

# Reference reward + data from the CUDA verl leg (reuse as-is).
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  data.image_key=images \
  data.image_patch_size=16 \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=cholec-sft-qwen3.5-4b \
  actor_rollout_ref.rollout.name=naive \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  reward.custom_reward_function.path="$REPO/verl/scripts/reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  "$@"

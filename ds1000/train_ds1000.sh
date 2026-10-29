#!/usr/bin/env bash
# DS-1000 RLVR (GRPO) on verl + vLLM rollout + TorchTPU, 1 epoch.
set -euo pipefail

export VERL_PLATFORM=tpu
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin   # official PlatformTPU
export TOKENIZERS_PARALLELISM=false

PY=~/.venv-verl-ds1000/bin/python
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
TRAIN_FILE=~/cholec/ds1000_parquet/train.parquet
VAL_FILE=~/cholec/ds1000_parquet/val.parquet
REWARD=~/ds1000/ds1000_reward.py
OUT_DIR=~/ds1000/runs

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.train_batch_size=16
  data.max_prompt_length=2048
  data.max_response_length=1024
  data.filter_overlong_prompts=True
  data.truncation='error'
)

REWARD_ARGS=(
  reward.custom_reward_function.path="$REWARD"
  reward.custom_reward_function.name=compute_score
)

MODEL=(
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.04
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.use_dynamic_bsz=False
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=0.7
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.response_length=1024
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.free_cache_engine=True
)

TRAINER=(
  trainer.critic_warmup=0
  trainer.logger='["console"]'
  trainer.project_name=ds1000_verl
  trainer.experiment_name=qwen3_vl_8b_ds1000_grpo
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.val_before_train=True
  trainer.save_freq=-1
  trainer.test_freq=16
  trainer.total_epochs=1
  trainer.resume_mode=disable
)

"$PY" -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${REWARD_ARGS[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${TRAINER[@]}" \
  "$@"

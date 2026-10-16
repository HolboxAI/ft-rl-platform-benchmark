#!/usr/bin/env bash
# CholecT50 surgical triplet recognition — GRPO (RLVR) with the triplet-F1 reward.
#
# Matches the Tinker RL run: 256 prompts, batch_size=16, group_size=8 rollouts,
# lr=1e-5, max_tokens=256, thinking disabled. Runs on SageMaker with the SFT
# checkpoint mounted at /opt/ml/input/data/model and verl-format parquet at
# /opt/ml/input/data/data.
set -xeuo pipefail

# No network on the training node: resolve everything from the image's 16 GB uv
# cache and never touch the HF hub (the model is fully local).
export UV_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/opt/ml/input/data/model
TRAIN_FILE=/opt/ml/input/data/data/train.parquet
VAL_FILE=/opt/ml/input/data/data/val.parquet
OUT_DIR=/opt/ml/model

echo "=== model files ==="; ls -la "$MODEL_PATH" | head -30
echo "=== data files ===";  ls -la /opt/ml/input/data/data | head -30
echo "=== nvidia-smi ==="; nvidia-smi --query-gpu=name,memory.total --format=csv || true

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.image_key=images
  data.image_patch_size=16
  data.train_batch_size=16
  data.max_prompt_length=2048
  data.max_response_length=256
  data.filter_overlong_prompts=True
  data.truncation='error'
  +data.apply_chat_template_kwargs.enable_thinking=False
)

REWARD=(
  reward.custom_reward_function.path=/workspace/verl/cholec_reward.py
  reward.custom_reward_function.name=compute_score
)

MODEL=(
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.actor.optim.lr=1e-5
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.04
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.fsdp_config.fsdp_size=4
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
  actor_rollout_ref.actor.fsdp_config.offload_policy=False
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.ref.fsdp_config.param_offload=False
  actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
  actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=0.7
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.prompt_length=2048
  actor_rollout_ref.rollout.response_length=256
  actor_rollout_ref.rollout.max_model_len=4096
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.rollout.max_num_seqs=128
  actor_rollout_ref.rollout.ignore_eos=False
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256
)

TRAINER=(
  trainer.critic_warmup=0
  trainer.logger='["console"]'
  trainer.project_name=verl_cholec_grpo
  trainer.experiment_name=cholec_rl_qwen3_5_4b
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.balance_batch=True
  trainer.default_local_dir="$OUT_DIR"
  trainer.val_before_train=True
  trainer.save_freq=16
  trainer.test_freq=16
  trainer.total_epochs=1
  trainer.resume_mode=disable
)

LAUNCH=(uv run --frozen --offline --all-packages --extra vllm --extra fsdp python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --offline --all-packages --extra vllm --extra fsdp")

# Pre-flight gate: VERL_CFG_ONLY=1 resolves the full hydra config (all overrides)
# and exits before any model/data loading. Catches override typos and
# struct-locked new-key errors (the `+`-prefix class) before a real launch.
CFG_ONLY=()
if [[ "${VERL_CFG_ONLY:-0}" == "1" ]]; then
  CFG_ONLY=(--cfg job)
fi

"${LAUNCH[@]}" -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${REWARD[@]}" \
  "${ACTOR[@]}" \
  "${REF[@]}" \
  "${ROLLOUT[@]}" \
  "${TRAINER[@]}" \
  "${RAY[@]}" \
  "${CFG_ONLY[@]}"

echo "TRAINING_DONE exit=$?"

#!/usr/bin/env bash
# CholecT50 GRPO/RLVR on verl + Cloud TPU (TorchTPU), LoRA r=16.
#
# Replicates the Tinker RL stage (LoRA r=16, 256/300, triplet-F1 reward) on a
# single v6e-8 using verl. The reference verl CUDA run is
# ft-rl-platform-benchmark/verl/scripts/train.sh; the [TPU] lines below are the
# only intended differences (device, rollout engine, LoRA, strategy).
set -euo pipefail

export VERL_PLATFORM=tpu
export VERL_ENGINE_DEVICE=cuda              # [TPU] select the CUDA-registered FSDP2 engine; it still uses `tpu` internally
export VERL_ENGINE_VENDOR=nvidia
export TORCH_DEVICE_BACKEND_AUTOLOAD=0     # [TPU] one process per chip; no autoload
export USE_HUB_KERNELS=NO                  # [TPU] pure-PyTorch Gated DeltaNet
export TOKENIZERS_PARALLELISM=false
PY=~/.venv-verl-tpu/bin/python
export PATH="$HOME/.venv-verl-tpu/bin:$PATH"   # ray CLI + venv tools

MODEL_PATH=~/models/cholec-sft-qwen3.5-4b
TRAIN_FILE=~/cholec/verl_parquet/train.parquet
VAL_FILE=~/cholec/verl_parquet/val.parquet
OUT_DIR=~/verl-tpu/runs/cholec

# Start Ray with one "GPU" slot per TPU chip (4 actor chips + 1 ref chip + 1 rollout chip).
ray stop --force >/dev/null 2>&1 || true
ray start --head --num-gpus=6 --disable-usage-stats >/dev/null 2>&1

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
  reward.custom_reward_function.path="$HOME/verl-tpu/cholec_reward.py"
  reward.custom_reward_function.name=compute_score
)

MODEL=(
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  # LoRA r=16, matching Tinker (train_attn + train_mlp + train_unembed).
  actor_rollout_ref.model.lora_rank=16
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.lora.merge=True     # [TPU] merge LoRA into base for the HF rollout
)

ACTOR=(
  actor_rollout_ref.actor.strategy=fsdp2      # [TPU] FSDP2 is proven on TorchTPU (FSDP1 is not)
  actor_rollout_ref.actor.optim.lr=1e-5
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.04
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.use_torch_compile=False
)

REF=(
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=hf           # [TPU] vLLM does not run on TPU; use HF generate
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=0.7
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.prompt_length=2048
  actor_rollout_ref.rollout.response_length=256
  actor_rollout_ref.rollout.ignore_eos=False
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.free_cache_engine=True
)

TRAINER=(
  trainer.critic_warmup=0
  trainer.use_v1=False                       # [TPU] V1 hard-requires transfer_queue (Ascend); use V0
  trainer.logger='["console"]'
  trainer.project_name=verl_cholec_tpu
  trainer.experiment_name=cholec_rl_qwen3_5_4b_loratpu
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

"$PY" -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${REWARD[@]}" \
  "${ACTOR[@]}" \
  "${REF[@]}" \
  "${ROLLOUT[@]}" \
  "${TRAINER[@]}" \
  "$@"

echo "TRAINING_DONE exit=$?"

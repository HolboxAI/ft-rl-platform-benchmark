#!/usr/bin/env bash
# CholecT50 GRPO (RLVR) on SkyRL — mirrors the verl run exactly:
#   Qwen3.5-4B (SFT checkpoint, thinking-disabled), 256/32 CholecT50 partition,
#   GRPO: 16 prompts/batch, 8 rollouts/prompt, lr 1e-5, kl 0.04, temp 0.7,
#   max_tokens 256, max_prompt 4096, 1 epoch, FSDP bf16 (4x A10G).
#
# Smoke test: pass `trainer.max_training_steps=1` as an extra arg.
set -x

# uv installs to ~/.local/bin, which non-login SSH shells don't put on PATH.
export PATH="$HOME/.local/bin:$PATH"

# FSDP2 `cpu_offload=true` pins params + fp32 AdamW states + master weights to
# CPU (CPUOffloadPolicy(pin_memory=True)), peaking ~180GB on 4xA10G vs 186GB RAM.
# Raise Ray's OOM-kill threshold so the peak doesn't trigger the killer; a 32G
# swapfile (added by setup.sh) absorbs non-pinned spillover.
export RAY_memory_usage_threshold=0.99

# transformers 5.16.1 defaults to "hub kernels" (kernels-community/fla +
# kernels-community/mamba-ssm) for the Qwen3.5 gated-delta-net (linear-attention)
# layers; those precompiled kernels emit garbage on A10G (sm86). Force the
# fallback to the installed fla + causal_conv1d packages instead — the exact
# kernel path verl's working run uses.
export USE_HUB_KERNELS=NO

# Resolve the SkyRL repo root (this script lives at examples/train/cholec/,
# three levels below the repo root).
SKYRL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "$SKYRL_ROOT"

: "${DATA_DIR:="$HOME/cholec/skyrl_parquet"}"
: "${MODEL_PATH:="$HOME/cholec/merged"}"
: "${NUM_GPUS:=4}"
: "${EXPORT_PATH:="$HOME/exports/cholec_skyrl"}"
: "${CKPT_PATH:="$HOME/ckpts/cholec_skyrl"}"

uv run --isolated --extra fsdp \
  python examples/train/cholec/entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/val.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.04 \
  trainer.algorithm.use_entropy_loss=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=16 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=16 \
  trainer.remove_microbatch_padding=false \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=256 \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=1.0 \
  generator.max_turns=1 \
  trainer.policy.optimizer_config.lr=1e-5 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.vision_language_generator=true \
  environment.env_class=cholec \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.engine_init_kwargs.max_model_len=8192 \
  generator.inference_engine.max_num_seqs=64 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.bf16=true \
  trainer.logger=console \
  trainer.project_name=cholec \
  trainer.run_name=cholec_rl_qwen3_5_4b_skyrl \
  trainer.resume_mode=none \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  "$@"

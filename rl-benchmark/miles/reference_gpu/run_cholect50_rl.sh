#!/bin/bash
# CholecT50 GRPO (RLVR) on Miles — strict reproduction of the verl reference.
#
# verl: FSDP actor (fsdp_size 4), FSDP2 reference, vLLM rollout, batch 16, n=8,
# lr 1e-5, max_prompt 2048, max_response 256, KL via loss (coef 0.04, low_var_kl),
# KL not in reward, thinking disabled, 1 epoch = 14 steps, val greedy mean@1.
#
# Miles: FSDP backend (matches verl FSDP + offload via --colocate), SGLang rollout.
#
# Runs INSIDE radixark/miles:latest. Assumes:
#   - /root/models/cholec-sft-qwen3.5-4b/   HF weights (synced from S3)
#   - /root/miles_bench/                    reward_cholect50.py, generate_cholect50.py,
#                                           log_rollout_cholect50.py,
#                                           cholect50_train.jsonl, cholect50_val.jsonl
set -ex

MODEL_NAME=${MODEL_NAME:-cholec-sft-qwen3.5-4b}
NUM_GPUS=${NUM_GPUS:-4}
BENCH_DIR=${BENCH_DIR:-/root/miles_bench}

export PYTHONUNBUFFERED=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export no_proxy="127.0.0.1,localhost"

# Clean stale workers.
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 miles 2>/dev/null || true
sleep 3

# FA4 crashes on Qwen3.5's 16/4 GQA layout. Uninstall so the trainer falls back to
# FA2 (same fix as the ORena run).
python3 -m pip uninstall -y flash-attn-4 2>/dev/null || true

# train.py and the `miles` package live in the repo root (/root/miles in the image).
cd /root/miles

ray start --head \
  --node-ip-address ${MASTER_ADDR} \
  --num-gpus ${NUM_GPUS} \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

CKPT_ARGS=(
  --hf-checkpoint "/root/models/${MODEL_NAME}"
)

ROLLOUT_ARGS=(
  --prompt-data "${BENCH_DIR}/cholect50_train.jsonl"
  --input-key messages
  --label-key label
  --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --rollout-shuffle
  --custom-rm-path reward_cholect50.reward_fn
  --reward-key reward_value
  --custom-generate-function-path generate_cholect50.generate
  --custom-rollout-log-function-path log_rollout_cholect50.log_rollout_data
  # --num-rollout counts steps, not rollouts. One epoch over 235 effective rows at
  # batch 16 is 14 steps, matching verl.
  --num-epoch 1
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --rollout-max-response-len 256
  --rollout-temperature 0.7
  --rollout-top-p 1.0
  --global-batch-size 128
)

EVAL_ARGS=(
  --eval-interval 16
  --eval-prompt-data cholect50_val "${BENCH_DIR}/cholect50_val.jsonl"
  --n-samples-per-eval-prompt 1
  --eval-max-response-len 256
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.04
  --kl-loss-type low_var_kl
  # verl has use_kl_in_reward=False: KL only in the loss, not in the reward.
  --kl-coef 0.00
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-5
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

BACKEND_ARGS=(
  --train-backend fsdp
  --gradient-checkpointing
  # Offload params+grads+optimizer to CPU. This is verl's param_offload=True +
  # optimizer_offload=True equivalent.
  --fsdp-cpu-offload
  # Keep SGLang weights resident on GPU; offload only KV cache during training.
  # This is verl's vLLM behavior (vLLM frees only KV cache, never the weights).
  --offload-rollout-level kv_cache
  # Attention backend for SGLang (rollout). Tuned at the smoke test for Qwen3.5
  # GDN linear attention; triton is the A10G-safe default from the de-risk leg.
  --sglang-attention-backend triton
  --update-weight-buffer-size 536870912
)

SGLANG_ARGS=(
  # TP=1 matches verl's vLLM default (tensor_parallel_size unset => 1). On 4xA100-40GB
  # the resident SGLang weights fit comfortably, so no TP split is needed.
  --rollout-num-gpus-per-engine 1
  # verl uses vLLM gpu_memory_utilization 0.6.
  --sglang-mem-fraction-static 0.6
)

SAVE_ARGS=(
  --save "${BENCH_DIR}/ckpt_cholect50"
  --save-interval 16
)

MISC_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node ${NUM_GPUS}
  # colocate = SGLang + FSDP on the same GPUs (its "offload" is only time-slicing,
  # not param offload). --fsdp-cpu-offload above is the real verl match.
  --colocate
)

# NOTE: do NOT add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. It is
# inherited by the SGLang cells, whose --enable-memory-saver crashes on it
# ("TorchMemorySaver ... expandable_segments is not supported yet"). verl uses
# expandable_segments only because it runs vLLM. The OOM is fixed by
# --fsdp-cpu-offload instead.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${BENCH_DIR}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

cd /root/miles
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${SAVE_ARGS[@]}" \
  "${MISC_ARGS[@]}"

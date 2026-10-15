#!/bin/bash
# ORena VLM RL (GRPO) on Miles — full 2000-rollout comparable run.
#
# Mirrors examples/geo3k_vlm/run_geo3k_vlm.sh (the canonical Qwen3-VL-8B-Instruct
# VLM RL path) with our data + reward + rollout hyperparameters. This is the
# apples-to-apples leg against Fireworks' ORena RL: same base model
# (Qwen/Qwen3-VL-8B-Instruct), same 2000 rows, same 8 samples/prompt, same
# 512-token cap, same temperature 0.7, same kl 0.04, same exact-match reward.
#
# Runs INSIDE the radixark/miles:latest container. Assumes:
#   - /root/models/Qwen3-VL-8B-Instruct/  HF weights (downloaded)
#   - /root/miles_bench/                  reward_orena.py, generate_orena.py,
#                                         orena_rl_miles.jsonl
set -ex

MODEL_NAME=${MODEL_NAME:-Qwen3-VL-8B-Instruct}
NUM_GPUS=${NUM_GPUS:-8}
BENCH_DIR=${BENCH_DIR:-/root/miles_bench}

export PYTHONUNBUFFERED=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export no_proxy="127.0.0.1,localhost"

# Detect NVLink (A100 has NVLink; the env var gates NCCL NVLS).
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Download model (idempotent).
mkdir -p /root/models
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
  hf download "Qwen/${MODEL_NAME}" --local-dir "/root/models/${MODEL_NAME}"
fi

# FA4 (flash-attn-4 4.0.0b19) is bundled in the image and, on SM80 (A100), is
# selected OVER FA2 (flash-attn 2.7.4.post1) by transformer_engine's backend
# ranking. FA4's CUTLASS GQA-packing kernel cannot compile Qwen3-VL-8B's 32 Q-head
# / 8 KV-head layout ("unable to compute crd2idx ... {i64 div=8}" in
# flash_attn/cute/pack_gqa.py). SGLang (rollout) does not import FA4 at all — only
# transformer_engine (Megatron training) does — so uninstall it and let TE fall back
# to FA2, which is stable on A100 and supports GQA. Idempotent; survives restarts.
python3 -m pip uninstall -y flash-attn-4 2>/dev/null || true

# Clean any stale workers.
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 miles 2>/dev/null || true
sleep 3

# Common checkpoint args. --hf-checkpoint feeds the tokenizer/processor and is
# the FSDP-style weight path; megatron also gets --load below (bridge mode).
CKPT_ARGS=(
  --hf-checkpoint "/root/models/${MODEL_NAME}"
)

ROLLOUT_ARGS=(
  --prompt-data "${BENCH_DIR}/orena_rl_miles.jsonl"
  --input-key messages
  --label-key label
  --apply-chat-template
  # Qwen3-VL defaults to thinking mode; without this the model fills the 512-token
  # budget with <think> reasoning and parses to zero reward (the exact failure
  # SageMaker's ORena RL hit, 0.0). Fireworks disabled thinking too.
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --rollout-shuffle
  --custom-rm-path reward_orena.reward_fn
  --reward-key reward_value
  --log-reward-category reward_cat
  --custom-generate-function-path generate_orena.generate
  # Log exact_match (strict 0/1) alongside reward_value (dense correctness). The
  # latter is the GRPO training signal; the former is the number Fireworks reports
  # (0.1925 -> 0.3200) and what the comparison plot needs.
  --custom-rollout-log-function-path log_rollout_orena.log_rollout_data
  # IMPORTANT: --num-rollout counts rollout *steps* (each draining --rollout-batch-size
  # 64 prompts), NOT total rollouts. "2000 rollouts" (== Fireworks' 2000 unique rows,
  # 1 epoch) is --num-epoch 1 => 2000 // 64 = 31 steps. (--num-rollout 2000 would be
  # 64 epochs / ~49 h, which is neither intended nor comparable.)
  --num-epoch 1
  --rollout-batch-size 64
  --n-samples-per-prompt 8
  --rollout-max-response-len 512
  --rollout-temperature 0.7
  --global-batch-size 512
)

# required for VLM datasets
MULTIMODAL_KEYS='{"image": "images"}'

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.04
  --kl-loss-type low_var_kl
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

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.6
  --sglang-cuda-graph-bs-decode 1 2 4 8 16 24 32 40 48 56 64
)

SAVE_ARGS=(
  --save "${BENCH_DIR}/ckpt_orena"
  # ~16 GB/checkpoint (8B params, no optimizer) -> 500 gives 4 checkpoints on the
  # 400 GB root volume, leaving headroom next to the 75 GB docker image + model.
  --save-interval 500
  --no-save-optim
)

MISC_ARGS=(
  --colocate
)

# Megatron backend (matches the geo3k default). --load points at the HF dir;
# --megatron-to-hf-mode bridge auto-converts HF -> Megatron on load.
BACKEND_ARGS=(
  --train-backend megatron
  --load "/root/models/${MODEL_NAME}"
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  # --attention-backend flash. With flash-attn-4 uninstalled above, this resolves to
  # FA2 (flash-attn 2.7.4.post1): stable on A100, supports GQA, and matches the
  # canonical geo3k_vlm example. (Do NOT use --attention-backend fused: cuDNN fused
  # attention is also rejected for Qwen3-VL-8B's GQA layout on sm80.)
  --attention-backend flash
  --megatron-to-hf-mode bridge
)

# Model-specific Megatron args (hidden size, layers, heads, GQA, rotary-base).
# VL models require rotary-base 5000000 (vs the text default 1e6).
# The miles repo is baked into the image at /root/miles (not under BENCH_DIR).
MILES_DIR=/root/miles
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
MODEL_ARGS_LINE="$(MODEL_ARGS_ROTARY_BASE=5000000 python3 "${MILES_DIR}/miles/utils/external_utils/model_args_utils.py" "${MODEL_ARGS_FILE}")" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
echo "MODEL_ARGS_FILE: ${MODEL_ARGS_FILE}"
echo "MODEL_ARGS: ${MODEL_ARGS_LINE}"

# Start Ray from the repo root. Ray runs the submitted job in the head node's
# working directory, so train.py must resolve to /root/miles/train.py (the repo
# root) — start ray AND submit from here.
cd /root/miles
ray start --head \
  --node-ip-address ${MASTER_ADDR} \
  --num-gpus ${NUM_GPUS} \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${BENCH_DIR}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

cd /root/miles
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${NUM_GPUS} \
  --multimodal-keys "${MULTIMODAL_KEYS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${SAVE_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${MISC_ARGS[@]}"

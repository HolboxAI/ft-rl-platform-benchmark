#!/bin/bash
# Phase 1 on TPU: CholecT50 GRPO, Qwen3.5-4B VLM from the merged SFT checkpoint,
# LoRA rank 16 (comparable to the Tinker RL leg). Start after run_smoke.sh passes.
#
#   bash common/fetch_cholect50.sh                                   # once (needs AWS read access)
#   bash torchtpu/scripts/run_cholect50_rl.sh --num-steps 1          # TorchTPU, short check
#   COMPILE=1 bash torchtpu/scripts/run_cholect50_rl.sh              # TorchTPU + torch.compile
#   TPU_PATH=torchxla bash torchtpu/scripts/run_cholect50_rl.sh      # PyTorch/XLA baseline
set -euo pipefail
cd "$(dirname "$0")/../.."
source torchtpu/scripts/tpu_env.sh

SFT_CKPT=${SFT_CKPT:-$HOME/models/cholec-sft-qwen3.5-4b}
TAG=${TPU_PATH}$([ "${COMPILE:-0}" = "1" ] && echo "-compile")
RUN_ID=${RUN_ID:-cholect50-rl-${TAG}-$(date -u +%Y%m%d-%H%M%S)}
OUT=torchtpu/logs/${RUN_ID}
mkdir -p "$OUT"
export JOB_ID=${RUN_ID}
export HOURLY_USD=${HOURLY_USD:-0}   # set to the slice's on-demand price (e.g. v6e-1)

python -m common.grpo.train_grpo \
  --backend "$BACKEND" --platform "torchtpu-${TAG}" --task cholect50 \
  --model "$SFT_CKPT" \
  --rollout-batch-size 16 --n-samples 8 --temperature 0.7 \
  --max-prompt-len 2048 --max-new-tokens 256 \
  --lr 1e-5 --kl-beta 0.04 \
  --lora-rank "${LORA_RANK:-16}" --dtype bf16 --grad-ckpt \
  --gen-micro-batch 8 --train-micro-batch 1 \
  --eval-split val \
  $([ "${COMPILE:-0}" = "1" ] && echo "--compile") \
  --out-dir "$OUT" "$@" 2>&1 | tee "$OUT/console.log"

cp "$OUT/final_results.json" "torchtpu/metrics/cholect50_rl_${TAG}_final_results.json"
cp "$OUT/steps.jsonl" "torchtpu/metrics/cholect50_rl_${TAG}_steps.jsonl"

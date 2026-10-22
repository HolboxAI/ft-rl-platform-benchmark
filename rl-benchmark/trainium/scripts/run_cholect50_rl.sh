#!/bin/bash
# Phase 1 on Trainium: CholecT50 GRPO, Qwen3.5-4B VLM from the merged SFT checkpoint.
# Start after run_smoke.sh passes. Not yet run on hardware: expect porting work.
#
# LoRA rank 16 on one device (comparable to the Tinker RL leg). Full-model RL
# (comparable to verl / Miles) needs sharding across NeuronCores, which this
# single-device trainer does not do.
#
#   bash common/fetch_cholect50.sh                          # once: checkpoint + parquet
#   bash trainium/scripts/run_cholect50_rl.sh --num-steps 1 # short check
#   bash trainium/scripts/run_cholect50_rl.sh               # full: 16 steps
set -euo pipefail
cd "$(dirname "$0")/../.."
source trainium/scripts/neuron_env.sh

SFT_CKPT=${SFT_CKPT:-$HOME/models/cholec-sft-qwen3.5-4b}
RUN_ID=${RUN_ID:-cholect50-rl-$(date -u +%Y%m%d-%H%M%S)}
OUT=trainium/logs/${RUN_ID}
mkdir -p "$OUT"
export JOB_ID=${RUN_ID}
export HOURLY_USD=${HOURLY_USD:-1.34}   # trn1.2xlarge on-demand, us-east-1 — verify; set for larger instances

python -m common.grpo.train_grpo \
  --backend neuron --platform trainium --task cholect50 \
  --model "$SFT_CKPT" \
  --rollout-batch-size 16 --n-samples 8 --temperature 0.7 \
  --max-prompt-len 2048 --max-new-tokens 256 \
  --lr 1e-5 --kl-beta 0.04 \
  --lora-rank "${LORA_RANK:-16}" --dtype bf16 --grad-ckpt \
  --gen-micro-batch 8 --train-micro-batch 1 \
  --eval-split val \
  --out-dir "$OUT" "$@" 2>&1 | tee "$OUT/console.log"

cp "$OUT/final_results.json" trainium/metrics/cholect50_rl_final_results.json
cp "$OUT/steps.jsonl" trainium/metrics/cholect50_rl_steps.jsonl

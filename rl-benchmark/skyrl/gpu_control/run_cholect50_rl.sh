#!/bin/bash
# CholecT50 GRPO, GPU control (FSDP backend). Same client as the TPU / Trainium runs.
#
#   PRESET=tinker     bash skyrl/gpu_control/run_cholect50_rl.sh   # Tinker-leg recipe (LoRA 16, IS loss)
#   PRESET=skyrl-grpo bash skyrl/gpu_control/run_cholect50_rl.sh   # closest to the existing SkyRL run
#
# skyrl-grpo trains the full model: restart the server between presets so each
# run starts from the untouched SFT weights. Extra flags pass through, e.g.
#   PRESET=skyrl-grpo bash ... --num-substeps 1      # ablation: one update per step
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv-skyrl-client/bin/activate
MODEL=${MODEL:-$HOME/models/cholec-sft-qwen3.5-4b}
PRESET=${PRESET:-tinker}
RUN_ID=${RUN_ID:-cholect50-rl-gpu-${PRESET}-$(date -u +%Y%m%d-%H%M%S)}
OUT=skyrl/logs/$RUN_ID
mkdir -p "$OUT"
export HOURLY_USD=${HOURLY_USD:-5.67}   # g5.12xlarge on-demand, us-east-1 — verify
TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy} python skyrl/scripts/cholect50_rl_client.py \
  --base-url "http://localhost:${PORT:-8000}" --model "$MODEL" --preset "$PRESET" \
  --platform "skyrl-gpu-fsdp-${PRESET}" --out-dir "$OUT" "$@" 2>&1 | tee "$OUT/console.log"
cp "$OUT/final_results.json" "skyrl/metrics/cholect50_rl_gpu_${PRESET}_final_results.json"

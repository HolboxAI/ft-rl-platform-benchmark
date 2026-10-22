#!/bin/bash
# CholecT50 GRPO via SkyRL on TPU. Only meaningful once run_probe.sh passes the
# image_input stage on the SFT checkpoint — today the JAX backend's Qwen3.5 is
# text-only, so expect the first rollout to fail. The failure is the result
# until that changes; record it in skyrl/config/SKYRL_CHOLECT50_CONFIG.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv-skyrl-client/bin/activate
MODEL=${MODEL:-$HOME/models/cholec-sft-qwen3.5-4b}
RUN_ID=${RUN_ID:-cholect50-rl-tpu-$(date -u +%Y%m%d-%H%M%S)}
OUT=skyrl/logs/$RUN_ID
mkdir -p "$OUT"
TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy} python skyrl/scripts/cholect50_rl_client.py \
  --base-url "http://localhost:${PORT:-8000}" --model "$MODEL" \
  --platform skyrl-tpu-jax --out-dir "$OUT" "$@" 2>&1 | tee "$OUT/console.log"
cp "$OUT/final_results.json" skyrl/metrics/cholect50_rl_tpu_final_results.json

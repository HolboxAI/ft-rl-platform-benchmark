#!/bin/bash
# SkyRL Tinker-API server on a Cloud TPU VM, JAX backend.
#
#   MODEL=Qwen/Qwen3-0.6B bash skyrl/tpu/start_server.sh                    # stage 1-4 probe
#   MODEL=$HOME/models/cholec-sft-qwen3.5-4b bash skyrl/tpu/start_server.sh  # CholecT50
#
# Ready when the log shows "Uvicorn running on http://0.0.0.0:8000".
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SKYRL_DIR=${SKYRL_DIR:-$HOME/SkyRL}
MODEL=${MODEL:?set MODEL (HF id or local checkpoint dir)}
# One chip by default. On a v6e-4/-8 slice, shard the model:
#   BACKEND_CONFIG='{"fully_sharded_data_parallel_size": 4, "max_lora_rank": 16}'
BACKEND_CONFIG=${BACKEND_CONFIG:-'{"max_lora_rank": 16, "max_lora_adapters": 2, "gradient_checkpointing": true}'}
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$REPO/skyrl/logs"
cd "$SKYRL_DIR"
# --extra tpu pulls jax[tpu] (libtpu); the API server forwards its uv flags to
# the engine process, which adds --extra jax itself.
uv run --extra tpu --extra tinker -m skyrl.tinker.api \
  --base-model "$MODEL" --backend jax --backend-config "$BACKEND_CONFIG" \
  --port "${PORT:-8000}" --checkpoints-base "${CKPT_DIR:-/tmp/skyrl_checkpoints}" \
  2>&1 | tee "$REPO/skyrl/logs/server_tpu_$(date -u +%Y%m%d-%H%M%S).log"

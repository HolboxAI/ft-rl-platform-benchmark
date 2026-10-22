#!/bin/bash
# SkyRL Tinker-API server on Trainium: JAX backend on jax-neuronx (beta). Speculative.
#
#   MODEL=Qwen/Qwen3-0.6B bash skyrl/trainium/start_server.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SKYRL_DIR=${SKYRL_DIR:-$HOME/SkyRL}
MODEL=${MODEL:?set MODEL}
BACKEND_CONFIG=${BACKEND_CONFIG:-'{"max_lora_rank": 16, "max_lora_adapters": 2, "gradient_checkpointing": true}'}
export PATH="$HOME/.local/bin:$PATH"
# Neuron compile cache shared with the PyTorch leg.
export NEURON_COMPILE_CACHE_URL=${NEURON_COMPILE_CACHE_URL:-/var/tmp/neuron-compile-cache}
export NEURON_CC_FLAGS=${NEURON_CC_FLAGS:---model-type=transformer}

cd "$SKYRL_DIR"
# --no-sync keeps the jax-neuronx packages installed by setup_jax_neuron.sh; the
# server forwards this flag to the engine process.
uv run --no-sync -m skyrl.tinker.api \
  --base-model "$MODEL" --backend jax --backend-config "$BACKEND_CONFIG" \
  --port "${PORT:-8000}" --checkpoints-base "${CKPT_DIR:-/tmp/skyrl_checkpoints}" \
  2>&1 | tee "$REPO/skyrl/logs/server_trainium_$(date -u +%Y%m%d-%H%M%S).log"

#!/bin/bash
# Stage probe against the running TPU server (start_server.sh in another shell).
# Run it twice: once with the server on Qwen/Qwen3-0.6B (does the stack work at
# all?), once on the CholecT50 SFT checkpoint (does Qwen3.5-4B load, train,
# sample, and accept an image?).
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv-skyrl-client/bin/activate
MODEL=${MODEL:?set MODEL to the server\'s --base-model}
TAG=${TAG:-$(basename "$MODEL")}
TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy} python skyrl/scripts/probe_tinker_server.py \
  --base-url "http://localhost:${PORT:-8000}" --model "$MODEL" \
  --out "skyrl/logs/probe_tpu_${TAG}.json" "$@"

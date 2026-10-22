#!/bin/bash
# Stage probe against the running Trainium server. Same probe as TPU / GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv-skyrl-client/bin/activate
MODEL=${MODEL:?set MODEL to the server\'s --base-model}
TAG=${TAG:-$(basename "$MODEL")}
TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy} python skyrl/scripts/probe_tinker_server.py \
  --base-url "http://localhost:${PORT:-8000}" --model "$MODEL" \
  --out "skyrl/logs/probe_trainium_${TAG}.json" "$@"

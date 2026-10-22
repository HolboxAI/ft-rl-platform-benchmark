#!/bin/bash
# Runs ON the TPU VM: SkyRL setup, then the stage probe against a fresh JAX-backend
# server for each model in MODELS. Results -> skyrl/logs/probe_tpu_<model>.json.
#
#   MODELS="Qwen/Qwen3-0.6B" bash skyrl/tpu/run_all_probes.sh
#   MODELS="Qwen/Qwen3-0.6B $HOME/models/cholec-sft-qwen3.5-4b" bash skyrl/tpu/run_all_probes.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
MODELS=${MODELS:-Qwen/Qwen3-0.6B}
PORT=${PORT:-8000}
DEFAULT_BACKEND_CONFIG='{"max_lora_rank": 16, "max_lora_adapters": 2, "gradient_checkpointing": true}'
BACKEND_CONFIG=${BACKEND_CONFIG:-$DEFAULT_BACKEND_CONFIG}
export PATH="$HOME/.local/bin:$PATH" TINKER_API_KEY=tml-dummy PYTHONUNBUFFERED=1
mkdir -p skyrl/logs
exec > >(tee -a skyrl/logs/tpu_run_all_probes.log) 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }

[ -d .venv-skyrl-client ] || bash skyrl/scripts/setup_skyrl.sh
SKYRL_DIR=${SKYRL_DIR:-$HOME/SkyRL}
log "pre-building SkyRL TPU env"
(cd "$SKYRL_DIR" && uv sync --python 3.12 --extra tpu --extra tinker --extra jax) || log "uv sync failed"
(cd "$SKYRL_DIR" && uv run --no-sync python -c "import jax; print('jax', jax.__version__, jax.devices())") || log "jax device check failed"

for MODEL in $MODELS; do
  tag=$(basename "$MODEL")
  log "=== probe $MODEL"
  slog=skyrl/logs/server_tpu_${tag}.log
  ( cd "$SKYRL_DIR" && exec setsid uv run --no-sync --extra tpu --extra tinker -m skyrl.tinker.api \
      --base-model "$MODEL" --backend jax \
      --backend-config "$BACKEND_CONFIG" \
      --port "$PORT" ) > "$slog" 2>&1 &
  spid=$!
  up=0
  for i in $(seq 1 120); do
    kill -0 "$spid" 2>/dev/null || { log "server exited early"; tail -40 "$slog"; break; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/" || true)" != "000" ] && { up=1; log "server up after $((i*10)) s"; break; }
    sleep 10
  done
  if [ "$up" = 1 ]; then
    .venv-skyrl-client/bin/python skyrl/scripts/probe_tinker_server.py --base-url "http://localhost:$PORT" \
      --model "$MODEL" --out "skyrl/logs/probe_tpu_${tag}.json"
  fi
  kill -- -"$spid" 2>/dev/null; sleep 5; pkill -f skyrl.tinker 2>/dev/null; sleep 5
done
log "done; results in skyrl/logs/probe_tpu_*.json"

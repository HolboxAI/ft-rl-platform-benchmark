#!/usr/bin/env bash
# Start the rollout supervisor (which starts the server) on the ROLLOUT VM.
#
#   ENGINE=vllm bash ~/tpu_rl_benchmark/skyrl-tpu/rollout/start_rollout.sh       # foreground
#   ENGINE=hf DETACH=1 bash ~/tpu_rl_benchmark/skyrl-tpu/rollout/start_rollout.sh
#
# Server :8000 (SkyRL data + control plane), agent :8001 (weight reload).
# Miles: FRONT_PORT=30000 makes the supervisor serve SGLang's API there (proxy + /pull_weights +
# /update_weights_from_disk) instead of the agent API. Several engines on one host each need their own
# PORT, FRONT_PORT, ROLLOUT_CHIPS and TPU_PORT_BASE (miles-tpu/ops/start_rollout_engines.sh sets them).
# The first start compiles for TPU; later reloads reuse the compile caches below.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
ENGINE=${ENGINE:-vllm}
VENV=${VENV:-$HOME/.venv-rollout-$ENGINE}
MODEL=${MODEL:-$HOME/models/cholec-sft-qwen3.5-4b}
# SkyRL sends trainer.policy.model.path as the model name; any name is accepted, this is for /v1/models.
SERVED_NAME=${SERVED_NAME:-$MODEL}
PORT=${PORT:-8000}
LOG_DIR=${LOG_DIR:-$ROOT/logs/rollout}
TPU_PORT_BASE=${TPU_PORT_BASE:-8600}   # libtpu process port; TorchTPU slice builders at +10..
mkdir -p "$LOG_DIR"

export USE_HUB_KERNELS=NO TOKENIZERS_PARALLELISM=false

# Single-VM layout: pin the server (and every restart of it) to some chips of a host it
# shares with the trainer, e.g. ROLLOUT_CHIPS=6 on a v6e-8 whose chips 0-5 train.
# One process owns all of ROLLOUT_CHIPS (tensor parallel = number of chips).
if [ -n "${ROLLOUT_CHIPS:-}" ]; then
  eval "$(python3 - "$ROLLOUT_CHIPS" "$TPU_PORT_BASE" <<'EOF'
import shlex, sys
chips = [c for c in sys.argv[1].split(",") if c]
port_base = int(sys.argv[2])
shape = {1: "1,1,1", 2: "2,1,1", 4: "2,2,1", 8: "2,4,1"}[len(chips)]
env = {
    "TPU_VISIBLE_CHIPS": ",".join(chips),          # libtpu / JAX (vLLM-TPU)
    "TPU_CHIPS_PER_PROCESS_BOUNDS": shape,
    "TPU_PROCESS_BOUNDS": "1,1,1",
    "TPU_PROCESS_PORT": str(port_base),
    "TPU_PROCESS_ADDRESSES": f"localhost:{port_base}",
    "CLOUD_TPU_TASK_ID": "0",
    "TPU_VISIBLE_DEVICES": ",".join(chips),        # TorchTPU (HF engine)
    "TORCH_TPU_TOPOLOGY": shape,
    "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{port_base + 10 + i}" for i in range(len(chips))),
}
for k, v in env.items():
    print(f"export {k}={shlex.quote(v)}")
EOF
)"
  echo "rollout pinned to chips $ROLLOUT_CHIPS (TPU_VISIBLE_CHIPS=$TPU_VISIBLE_CHIPS)"
fi
# Persistent compile caches so each weight reload does not recompile from scratch.
export VLLM_XLA_CACHE_PATH=${VLLM_XLA_CACHE_PATH:-$HOME/.cache/vllm_xla}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax_compile}

case "$ENGINE" in
  # Matches the GPU run's vLLM settings where they apply (max_model_len 8192, max_num_seqs 64).
  vllm) SERVER_ARGS=(--engine vllm --max-model-len 8192 --max-num-seqs 64 --tensor-parallel-size 1
                     --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}") ;;
  hf)   SERVER_ARGS=(--engine hf --device tpu --max-batch "${MAX_BATCH:-16}" --len-bucket 128
                     --batch-wait-ms 300 ${CACHE_IMPL:+--cache-implementation "$CACHE_IMPL"}) ;;
  *) echo "ENGINE must be vllm or hf"; exit 1 ;;
esac

CMD=("$VENV/bin/python" "$HERE/rollout_supervisor.py" --base-model "$MODEL" --served-model-name "$SERVED_NAME"
     --server-port "$PORT" --work-dir "${WORK_DIR:-/tmp/rollout_weights}" --log-dir "$LOG_DIR"
     ${FRONT_PORT:+--front-port "$FRONT_PORT"} -- "${SERVER_ARGS[@]}")
echo "${CMD[*]}"
if [ "${DETACH:-0}" = 1 ]; then
  nohup setsid "${CMD[@]}" > "$LOG_DIR/supervisor.log" 2>&1 &
  echo "supervisor pid $!; log $LOG_DIR/supervisor.log; wait for: curl localhost:$PORT/health"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_DIR/supervisor.log"
fi

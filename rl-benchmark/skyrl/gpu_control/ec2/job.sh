#!/usr/bin/env bash
# SkyRL CholecT50 GPU control: the job the per-boot runner executes on the EC2 host (as root).
# Environment comes from control/job.env (written by submit.sh).
#
# Phases, each with a fresh SkyRL Tinker-API server (FSDP backend allows one trained
# model per server):
#   setup   uv, SkyRL @ SKYRL_REF (+ vision-RoPE patch hook), client venv
#   data    SFT checkpoint + CholecT50 parquet from S3
#   probe   probe_tinker_server.py incl. image input (validates the stack)
#   <preset> for each of PRESETS: cholect50_rl_client.py --preset <preset>
# Finished phases leave /data/state/<phase>.done, so a reboot skips them.
set -euo pipefail
: "${S3_ROOT:?}" "${RUN_ID:?}" "${MODEL_S3_URI:?}" "${PARQUET_S3_URI:?}" "${SKYRL_REF:?}"
PRESETS=${PRESETS:-"tinker skyrl-grpo"}

DATA=/data
CODE=$DATA/code/rl-benchmark
SKYRL=$DATA/SkyRL
CLIENT=$DATA/client-venv
MODEL=$DATA/models/cholec-sft-qwen3.5-4b
OUT=$DATA/runs/$RUN_ID
STATE=$DATA/state
S3_RUN=$S3_ROOT/runs/$RUN_ID
PORT=8000
export HOME=${HOME:-/root}
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=$DATA/uv-cache HF_HOME=$DATA/hf
# Ray's uv hook re-runs each worker as `uv run ...` inside a packaged copy of the SkyRL dir,
# which excludes .venv. Without this, uv creates an empty venv there and the workers fail with
# "No module named 'ray'" (run skyrl-gpu-20261022-1101). SkyRL forwards this var to workers.
export UV_PROJECT_ENVIRONMENT=$SKYRL/.venv
export USE_HUB_KERNELS=NO           # Qwen3.5 GDN hub kernels emit garbage on A10G (sm86); same as the SkyRL GPU leg
export TINKER_API_KEY=tml-dummy PYTHONUNBUFFERED=1
mkdir -p "$OUT" "$STATE" "$DATA/models"

log() { echo "[$(date -u +%FT%TZ)] $*"; }
stage() { echo "$1" > "$STATE/current_stage"; log "=== stage $1"; }
done_() { touch "$STATE/$1.done"; }
is_done() { [ -f "$STATE/$1.done" ]; }
sync_up() { aws s3 sync --only-show-errors "$OUT/" "$S3_RUN/out/" --exclude "*/checkpoints/*" || true; }

log "job $RUN_ID presets: $PRESETS"
aws s3 sync --delete --only-show-errors "$S3_ROOT/code/" "$CODE/"
( while sleep 300; do sync_up; done ) &
SYNCER=$!

server_pid=""
collect_logs() {  # $1 = phase. SkyRL's infra log + Ray worker errors live in /tmp, which is wiped on reboot.
  local d=$OUT/$1/infra
  mkdir -p "$d"
  cp /tmp/skyrl-logs/*.log "$d/" 2>/dev/null || true
  for f in /tmp/ray/session_latest/logs/*.err /tmp/ray/session_latest/logs/raylet.out; do
    [ -s "$f" ] && tail -c 2000000 "$f" > "$d/$(basename "$f")"
  done
  nvidia-smi > "$d/nvidia-smi.txt" 2>&1 || true
  free -g > "$d/free.txt" 2>&1 || true
}
stop_server() {
  [ -n "$server_pid" ] && collect_logs "$(cat "$STATE/current_stage" 2>/dev/null || echo unknown)"
  [ -n "$server_pid" ] && kill -- -"$server_pid" 2>/dev/null || true
  sleep 10
  pkill -f "skyrl.tinker" 2>/dev/null || true
  (cd "$SKYRL" && uv run --no-sync ray stop --force) >/dev/null 2>&1 || true
  pkill -f "vllm" 2>/dev/null || true
  sleep 15
  server_pid=""
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu after stop: /'
}
trap 'stop_server; kill $SYNCER 2>/dev/null || true; sync_up' EXIT

# backend-config shared by every phase: 4 GPUs colocated, CPU offload (as the reference run),
# VLM needs microbatch padding kept; vLLM limits from reference_gpu/run_cholec.sh.
BACKEND_CONFIG='{"trainer.remove_microbatch_padding": false,
 "trainer.placement.colocate_all": true,
 "trainer.placement.policy_num_gpus_per_node": 4,
 "trainer.placement.ref_num_gpus_per_node": 4,
 "generator.inference_engine.num_engines": 4,
 "generator.inference_engine.tensor_parallel_size": 1,
 "generator.inference_engine.gpu_memory_utilization": 0.6,
 "generator.inference_engine.engine_init_kwargs.max_model_len": 8192,
 "generator.inference_engine.max_num_seqs": 64,
 "trainer.policy.fsdp_config.cpu_offload": true}'

start_server() {  # $1 = phase name
  local slog=$OUT/$1/server.log
  mkdir -p "$OUT/$1"
  log "starting SkyRL server for $1"
  ( cd "$SKYRL" && exec setsid uv run --no-sync --extra tinker --extra fsdp -m skyrl.tinker.api \
      --base-model "$MODEL" --backend fsdp --backend-config "$BACKEND_CONFIG" \
      --port "$PORT" --checkpoints-base "$DATA/ckpts/$1" ) > "$slog" 2>&1 &
  server_pid=$!
  for i in $(seq 1 180); do   # up to 30 min (vLLM engine init + model load)
    if ! kill -0 "$server_pid" 2>/dev/null; then log "server exited early"; tail -50 "$slog"; return 1; fi
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/" || true)
    if [ "$code" != "000" ]; then log "server up after $((i * 10)) s (HTTP $code)"; return 0; fi
    sleep 10
  done
  log "server not up after 30 min"; tail -50 "$slog"; return 1
}

# ---------------------------------------------------------------- setup
if ! is_done setup; then
  stage setup
  nvidia-smi; free -g; df -h "$DATA"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  [ -d "$SKYRL/.git" ] || git clone https://github.com/NovaSky-AI/SkyRL "$SKYRL"
  git -C "$SKYRL" fetch --depth 1 origin "$SKYRL_REF" 2>/dev/null || git -C "$SKYRL" fetch origin
  git -C "$SKYRL" checkout "$SKYRL_REF"
  (cd "$SKYRL" && uv sync --python 3.12 --extra tinker --extra fsdp)
  # Vision-RoPE patch for every process in the venv (see qwen35_vision_rope_patch.py).
  SITE=$(cd "$SKYRL" && uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  cp "$CODE/skyrl/gpu_control/ec2/qwen35_vision_rope_patch.py" "$SITE/"
  echo "import qwen35_vision_rope_patch; qwen35_vision_rope_patch.install()" > "$SITE/zz_cholec_vision_rope.pth"
  (cd "$SKYRL" && uv run --no-sync python -c "import transformers.models.qwen3_5.modeling_qwen3_5 as m; assert getattr(m.Qwen3_5VisionRotaryEmbedding, '_cholec_patched', False); print('patch active')")
  uv venv --python 3.12 "$CLIENT"
  uv pip install --python "$CLIENT/bin/python" "tinker>=0.25,<0.28" "tinker-cookbook==0.5.7" pandas pyarrow pillow
  { echo "SkyRL $(git -C "$SKYRL" log -1 --format='%h %cs')"
    (cd "$SKYRL" && uv pip list 2>/dev/null | grep -iE "^(torch|vllm|transformers|ray|tinker|flash)")
    echo "--- client"; uv pip list --python "$CLIENT/bin/python" 2>/dev/null | grep -iE "^(tinker|torch|transformers)"
  } | tee "$OUT/environment.txt"
  done_ setup
fi

# ---------------------------------------------------------------- data
if ! is_done data || [ ! -f "$MODEL/config.json" ]; then
  stage data
  aws s3 sync --only-show-errors "$MODEL_S3_URI/" "$MODEL/"
  mkdir -p "$CODE/data/cholect50/parquet"
  aws s3 sync --only-show-errors "$PARQUET_S3_URI/" "$CODE/data/cholect50/parquet/"
  ls -la "$MODEL" "$CODE/data/cholect50/parquet" | tee "$OUT/data_manifest.txt"
  (cd "$CODE/data/cholect50/parquet" && sha256sum train.parquet val.parquet) | tee -a "$OUT/data_manifest.txt"
  done_ data
fi
# The client reads data/cholect50/parquet relative to the repo; keep it after a code re-sync.
[ -f "$CODE/data/cholect50/parquet/train.parquet" ] || aws s3 sync --only-show-errors "$PARQUET_S3_URI/" "$CODE/data/cholect50/parquet/"

# ---------------------------------------------------------------- probe
if ! is_done probe; then
  stage probe
  start_server probe
  "$CLIENT/bin/python" "$CODE/skyrl/scripts/probe_tinker_server.py" \
    --base-url "http://localhost:$PORT" --model "$MODEL" --out "$OUT/probe/probe.json" \
    2>&1 | tee "$OUT/probe/probe_console.log" || { log "probe FAILED: see probe.json"; stop_server; sync_up; exit 1; }
  stop_server
  done_ probe
  sync_up
fi

# ---------------------------------------------------------------- presets
for preset in $PRESETS; do
  is_done "preset-$preset" && { log "skip $preset (done)"; continue; }
  stage "preset-$preset"
  start_server "$preset" || { log "server for $preset failed to start"; stop_server; sync_up; continue; }
  t0=$(date +%s)
  "$CLIENT/bin/python" "$CODE/skyrl/scripts/cholect50_rl_client.py" \
    --base-url "http://localhost:$PORT" --model "$MODEL" --preset "$preset" \
    --data-dir "$CODE/data/cholect50/parquet" \
    --platform "skyrl-gpu-fsdp-$preset" --hourly-usd "${PRICE_PER_HOUR_USD:-0}" \
    --out-dir "$OUT/$preset" 2>&1 | tee "$OUT/$preset/client_console.log" \
    || { log "preset $preset FAILED after $(( $(date +%s) - t0 )) s"; stop_server; sync_up; continue; }
  log "preset $preset done in $(( $(date +%s) - t0 )) s"
  stop_server
  done_ "preset-$preset"
  sync_up
done

stage complete
log "job $RUN_ID complete"
for preset in $PRESETS; do [ -f "$STATE/preset-$preset.done" ] || exit 3; done

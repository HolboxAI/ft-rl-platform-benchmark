#!/usr/bin/env bash
# DS-1000 GRPO on ONE v6e-8: setup, data, model, reward check, rollout server and the run. Runs on that VM
# (started by startup_ds1000.sh at boot, or by hand):
#
#   nohup bash ~/tpu_rl_benchmark/skyrl-tpu/skyrl_ds1000/single_vm_ds1000.sh > ~/ds1000_single_vm.out 2>&1 &
#
# Chips as in the CholecT50 single-VM run (ops/single_vm.sh): policy 0-3, ref 4, HF rollout server on 6.
# The DS-1000 tests run on the host CPUs, in their own venv. Every stage is logged to
# logs/ds1000_single_vm.log and mirrored to $RUN_GCS/ds1000_single_vm/; the run's own logs go to
# $RUN_GCS/<run_id>/ (ops/watch_run.sh). Stops at the first failed stage.
#
# Env: MODEL_ID (default Qwen/Qwen3.5-4B), MAX_STEPS (20), EVAL_INTERVAL (10), RUN_ARGS (extra overrides,
#      e.g. "trainer.max_training_steps=1" for a smoke test).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
export TOPOLOGY=single
ROLLOUT_CHIPS=${ROLLOUT_CHIPS:-6}
RUN_GCS=${RUN_GCS:-gs://tpu-rl-run-cholec/runs}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-4B}
export MODEL_PATH=${MODEL_PATH:-$HOME/models/$(basename "$MODEL_ID")}
export MAX_STEPS=${MAX_STEPS:-20} EVAL_INTERVAL=${EVAL_INTERVAL:-10}
DS1000_REPO=${DS1000_REPO:-$HOME/DS-1000}
DATA_DIR=${DATA_DIR:-$HOME/ds1000/skyrl_parquet}
export DS1000_PYTHON=${DS1000_PYTHON:-$HOME/.venv-ds1000-exec/bin/python}
export DS1000_DATA_HOME=${DS1000_DATA_HOME:-$HOME/.cache/ds1000-data}
LOG="$ROOT/logs/ds1000_single_vm.log"
mkdir -p "$ROOT/logs"
export PATH="$HOME/.local/bin:$PATH"
PY=$HOME/.venv-skyrl-tpu/bin/python

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; gcloud storage cp -q "$LOG" "$RUN_GCS/ds1000_single_vm/single_vm.log" > /dev/null 2>&1; }
fail() { log "FAILED at stage: $1"; for f in "$ROOT"/logs/*.log "$ROOT"/logs/*.txt; do [ -f "$f" ] && gcloud storage cp -q "$f" "$RUN_GCS/ds1000_single_vm/" > /dev/null 2>&1; done; exit 1; }
stage() { log "== $1"; }

stage "setup: trainer venv (TorchTPU + SkyRL with the TPU patch)"
if [ -x "$PY" ] && "$PY" -c "import torch_tpu, skyrl" 2>/dev/null; then
  log "trainer venv present"
else
  bash "$ROOT/setup/setup_trainer_vm.sh" > "$ROOT/logs/setup_trainer.log" 2>&1 || fail "setup_trainer_vm.sh (logs/setup_trainer.log)"
fi

stage "setup: rollout venv (HF engine on TorchTPU)"
if [ -x "$HOME/.venv-rollout-hf/bin/python" ]; then
  log "rollout venv present"
else
  ENGINE=hf bash "$ROOT/setup/setup_rollout_vm.sh" > "$ROOT/logs/setup_rollout_hf.log" 2>&1 || fail "setup_rollout_vm.sh (logs/setup_rollout_hf.log)"
fi

stage "setup: DS-1000 test venv + offline sample data"
if [ -x "$DS1000_PYTHON" ] && [ -d "$DS1000_DATA_HOME/seaborn" ]; then
  log "test venv present"
else
  bash "$HERE/setup_exec_venv.sh" > "$ROOT/logs/setup_ds1000_exec.log" 2>&1 || fail "setup_exec_venv.sh (logs/setup_ds1000_exec.log)"
  log "test venv: $(grep -E '^(pandas|numpy|tensorflow|torch) ' "$ROOT/logs/setup_ds1000_exec.log" | tr -s ' ' | tr '\n' ' ')"
fi

stage "data: DS-1000 -> SkyRL parquet (train / val)"
# data.py and check_reward.py read parquet with pandas; the trainer venv may not have it
"$PY" -c "import pandas, pyarrow" 2>/dev/null || uv pip install -q --python "$PY" pandas pyarrow > /dev/null 2>&1 \
  || fail "pandas in the trainer venv"
if [ -f "$DATA_DIR/val.parquet" ]; then
  log "parquet present"
else
  [ -d "$DS1000_REPO" ] || git clone -q --depth 1 https://github.com/xlang-ai/DS-1000.git "$DS1000_REPO" || fail "clone DS-1000"
  "$PY" "$HERE/data.py" --src "$DS1000_REPO/data/ds1000.jsonl.gz" --out_dir "$DATA_DIR" > "$ROOT/logs/ds1000_data.log" 2>&1 \
    || fail "data.py (logs/ds1000_data.log)"
  log "DS-1000 @ $(git -C "$DS1000_REPO" rev-parse --short HEAD): $(grep wrote "$ROOT/logs/ds1000_data.log" | sed 's#.*/##' | tr '\n' ' ')"
fi

stage "model: $MODEL_ID"
if [ -f "$MODEL_PATH/config.json" ]; then
  log "model present"
else
  "$PY" - "$MODEL_ID" "$MODEL_PATH" > "$ROOT/logs/model_download.log" 2>&1 <<'EOF' || fail "model download (logs/model_download.log)"
import sys
from huggingface_hub import HfApi, snapshot_download
repo, out = sys.argv[1], sys.argv[2]
sha = HfApi().model_info(repo).sha
snapshot_download(repo, revision=sha, local_dir=out)
open(f"{out}/HF_REVISION", "w").write(f"{repo}@{sha}\n")
print("downloaded", repo, sha)
EOF
  log "model: $(cat "$MODEL_PATH/HF_REVISION")"
fi

stage "reward check on this VM (every DS-1000 reference passes, shortcuts fail)"
"$PY" "$HERE/check_reward.py" --ds1000_repo "$DS1000_REPO" --out_dir "$ROOT/logs/ds1000_check" \
  --checks reference,hacks --workers 32 > "$ROOT/logs/ds1000_check.log" 2>&1 || fail "check_reward.py (logs/ds1000_check.log)"
log "$(grep -E '^\[(PASS|FAIL)\] reference' "$ROOT/logs/ds1000_check.log")"
gcloud storage cp -q "$ROOT/logs/ds1000_check.log" "$RUN_GCS/ds1000_single_vm/" > /dev/null 2>&1

stage "rollout server on chip $ROLLOUT_CHIPS ($MODEL_ID, HF engine)"
if curl -sf -m 5 http://localhost:8000/health > /dev/null; then
  log "rollout server already healthy; reusing it"
else
  ENGINE=hf ROLLOUT_CHIPS=$ROLLOUT_CHIPS MODEL=$MODEL_PATH WORK_DIR=/dev/shm/rollout_weights DETACH=1 \
    bash "$ROOT/rollout/start_rollout.sh" >> "$LOG" 2>&1
  ok=0
  for i in $(seq 1 180); do  # the first TPU compile can take a while
    curl -sf -m 5 http://localhost:8000/health > /dev/null && { ok=1; break; }
    pgrep -f rollout_supervisor.py > /dev/null || break
    sleep 10
  done
  [ $ok = 1 ] || fail "rollout server (logs/rollout/)"
fi
log "rollout server healthy: $(curl -s -m 5 http://localhost:8000/version)"

stage "launch run + watcher (eval of the base model first, then $MAX_STEPS GRPO steps)"
export RUN_ID=${RUN_ID:-ds1000-skyrl-tpu-$(date -u +%Y%m%d-%H%M%S)}
RUN_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "$RUN_ID" > "$ROOT/logs/CURRENT_RUN"
# shellcheck disable=SC2086
nohup setsid bash -c '
  here=$0; dir=$1; shift
  bash "$here/run_ds1000_tpu.sh" "$@" > "$dir/launcher.log" 2>&1
  echo $? > "$dir/exit_code"
' "$HERE" "$RUN_DIR" ${RUN_ARGS:-} > /dev/null 2>&1 &
echo $! > "$RUN_DIR/run.pid"
EXPECTED_STEPS=$MAX_STEPS STALL_MIN=${STALL_MIN:-90} nohup setsid bash "$ROOT/ops/watch_run.sh" "$RUN_ID" > "$RUN_DIR/watcher.out" 2>&1 &
echo $! > "$RUN_DIR/watcher.pid"
log "run launched: $RUN_ID; progress in $RUN_GCS/$RUN_ID/"

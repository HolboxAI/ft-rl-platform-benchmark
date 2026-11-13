#!/usr/bin/env bash
# Start the CholecT50 run in the background on the TRAINER VM, plus its watcher.
# Both survive SSH disconnects and a closed laptop (nohup + setsid).
#
#   bash ~/tpu_rl_benchmark/skyrl-tpu/ops/launch_bg.sh                               # full run
#   bash ~/tpu_rl_benchmark/skyrl-tpu/ops/launch_bg.sh trainer.max_training_steps=1  # smoke
#
# Writes logs/<run_id>/{train.log,metrics.jsonl,exit_code,watch.log,status.json}
# and mirrors them to $RUN_GCS/<run_id>/ (see ops/watch_run.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID=${RUN_ID:-cholect50-skyrl-tpu-$(date -u +%Y%m%d-%H%M%S)}
export RUN_GCS=${RUN_GCS:-gs://tpu-rl-run-cholec/runs}
RUN_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "$RUN_ID" > "$ROOT/logs/CURRENT_RUN"

# The run, with its exit code recorded so the watcher can tell "finished" from "crashed".
nohup setsid bash -c '
  root=$0; dir=$1; shift
  bash "$root/skyrl_cholect50/run_cholect50_tpu.sh" "$@" > "$dir/launcher.log" 2>&1
  echo $? > "$dir/exit_code"
' "$ROOT" "$RUN_DIR" "$@" > /dev/null 2>&1 &
echo $! > "$RUN_DIR/run.pid"

nohup setsid bash "$ROOT/ops/watch_run.sh" "$RUN_ID" > "$RUN_DIR/watcher.out" 2>&1 &
echo $! > "$RUN_DIR/watcher.pid"

echo "run $RUN_ID started (pid $(cat "$RUN_DIR/run.pid")), watcher pid $(cat "$RUN_DIR/watcher.pid")"
echo "progress: tail -f $RUN_DIR/watch.log     or from the laptop: bash gcloud/run_status.sh"

#!/usr/bin/env bash
# Watcher for a background run. Runs on the TRAINER VM (started by ops/launch_bg.sh).
#
# Every $INTERVAL seconds it appends one line to logs/<run_id>/watch.log:
#   time | state | step n/16 | last train reward | last eval | step time | rollout health | last log line
# and rewrites status.json. Every $SYNC_EVERY ticks it mirrors watch.log, status.json,
# metrics.jsonl and the last 500 lines of train.log to $RUN_GCS/<run_id>/, so progress is
# readable from anywhere (gcloud/run_status.sh) even if this VM is preempted afterwards.
#
# States: STARTING -> RUNNING -> COMPLETED | CRASHED | STALLED (no new log output for $STALL_MIN min).
# On COMPLETED or CRASHED it records the error excerpt, builds report.html, uploads, and exits.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=${1:?run id}
RUN_DIR="$ROOT/logs/$RUN_ID"
RUN_GCS=${RUN_GCS:-gs://tpu-rl-run-cholec/runs}
INTERVAL=${INTERVAL:-60}
SYNC_EVERY=${SYNC_EVERY:-2}
STALL_MIN=${STALL_MIN:-45}
EXPECTED_STEPS=${EXPECTED_STEPS:-16}
ROLLOUT_URLS=${ROLLOUT_URLS:-$(grep -oE 'ROLLOUT_URLS:=[^}]*' "$ROOT/skyrl_cholect50/tpu_env.sh" | cut -d= -f2)}
PY=${PY:-$HOME/.venv-skyrl-tpu/bin/python}
WATCH="$RUN_DIR/watch.log"
tick=0

sync_gcs() {
  [ -f "$RUN_DIR/train.log" ] && tail -n 500 "$RUN_DIR/train.log" > "$RUN_DIR/train_tail.log"
  for f in watch.log status.json metrics.jsonl train_tail.log report.html final_results.json error_excerpt.txt exit_code environment.txt; do
    [ -f "$RUN_DIR/$f" ] && gcloud storage cp -q "$RUN_DIR/$f" "$RUN_GCS/$RUN_ID/$f" > /dev/null 2>&1
  done
}

summary() {  # step, reward, eval from metrics.jsonl (JSON on stdout)
  "$PY" - "$RUN_DIR/metrics.jsonl" "$EXPECTED_STEPS" <<'EOF'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
step, reward, ev, ev0, st = 0, None, None, None, None
try:
    for line in open(path):
        r = json.loads(line); d = r.get("data", {}); s = int(r.get("step", 0))
        if "reward/avg_raw_reward" in d: step, reward = max(step, s), d["reward/avg_raw_reward"]
        if "eval/all/avg_score" in d:
            ev = d["eval/all/avg_score"]
            if s == 0: ev0 = ev
        if "timing/step" in d: st = d["timing/step"]
except FileNotFoundError:
    pass
f = lambda v: None if v is None else round(float(v), 4)
print(json.dumps({"step": step, "expected_steps": expected, "train_reward": f(reward), "eval_before": f(ev0),
                  "eval_latest": f(ev), "step_seconds": f(st)}))
EOF
}

rollout_health() {
  local out=""
  IFS=, read -r -a urls <<< "$ROLLOUT_URLS"
  for u in "${urls[@]}"; do
    if curl -sf -m 5 "$u/health" > /dev/null 2>&1; then out+="ok "; else out+="DOWN "; fi
  done
  echo "${out% }"
}

error_excerpt() {
  grep -nE "Traceback|Error|error:|OOM|Killed|RESOURCE_EXHAUSTED|out of memory" "$RUN_DIR/train.log" "$RUN_DIR/launcher.log" 2>/dev/null | tail -n 40
}

echo "$(date -u +%FT%TZ) watcher started for $RUN_ID (interval ${INTERVAL}s, gcs $RUN_GCS/$RUN_ID)" >> "$WATCH"
state=STARTING
while true; do
  tick=$((tick + 1))
  now=$(date -u +%FT%TZ)
  pid=$(cat "$RUN_DIR/run.pid" 2>/dev/null || echo 0)
  alive=0; kill -0 "$pid" 2>/dev/null && alive=1
  sum=$(summary)
  step=$(echo "$sum" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['step'])")
  last_line=$(tail -n 1 "$RUN_DIR/train.log" 2>/dev/null | cut -c1-160 | tr -d '\r')
  log_age_min=$(( ( $(date +%s) - $(stat -c %Y "$RUN_DIR/train.log" 2>/dev/null || date +%s) ) / 60 ))
  health=$(rollout_health)

  if [ -f "$RUN_DIR/exit_code" ]; then
    code=$(cat "$RUN_DIR/exit_code")
    if [ "$code" = 0 ] && [ "$step" -ge "$EXPECTED_STEPS" ]; then state=COMPLETED
    elif [ "$code" = 0 ]; then state=COMPLETED_EARLY
    else state=CRASHED; fi
  elif [ "$alive" = 0 ]; then
    state=CRASHED; code="none (process gone without exit code)"
  elif [ "$log_age_min" -ge "$STALL_MIN" ]; then
    state=STALLED
  elif [ "$step" -gt 0 ]; then
    state=RUNNING
  fi

  printf '%s | %-15s | step %s/%s | %s | rollout %s | log age %smin | %s\n' \
    "$now" "$state" "$step" "$EXPECTED_STEPS" "$sum" "$health" "$log_age_min" "$last_line" >> "$WATCH"
  "$PY" - "$RUN_DIR/status.json" "$state" "$now" "$health" "$sum" "$log_age_min" <<'EOF'
import json, sys
path, state, now, health, summ, age = sys.argv[1:7]
json.dump({"state": state, "updated": now, "rollout": health, "log_age_min": int(age), **json.loads(summ)},
          open(path, "w"), indent=1)
EOF

  case "$state" in
    COMPLETED|COMPLETED_EARLY|CRASHED)
      echo "$now final state $state, exit code ${code:-?}" >> "$WATCH"
      if [ "$state" = CRASHED ]; then
        error_excerpt > "$RUN_DIR/error_excerpt.txt"
        echo "---- error excerpt ----" >> "$WATCH"; cat "$RUN_DIR/error_excerpt.txt" >> "$WATCH"
      fi
      [ -f "$RUN_DIR/metrics.jsonl" ] && "$PY" "$ROOT/analysis/plot_run.py" "$RUN_DIR" >> "$WATCH" 2>&1
      sync_gcs
      exit 0 ;;
  esac
  [ $((tick % SYNC_EVERY)) = 0 ] && sync_gcs
  sleep "$INTERVAL"
done

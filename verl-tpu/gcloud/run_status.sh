#!/usr/bin/env bash
# From the laptop (or anywhere with gcloud): how is the run doing?
#
#   bash gcloud/run_status.sh            # latest run: VM states, status.json, last 15 watcher lines
#   bash gcloud/run_status.sh <run_id>
#   bash gcloud/run_status.sh --report   # also download report.html when the run has one
#
# Reads what ops/watch_run.sh mirrors to GCS, so it works while the laptop was off and
# after a preemption. A PREEMPTED VM plus a status.json that stopped updating = the run died with the VM.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config/tpu.env.sh"
RUN_GCS=${RUN_GCS:-gs://tpu-rl-run-cholec/runs}
G=("$GCLOUD" --project "$PROJECT")
REPORT=0; RUN_ID=""
for a in "$@"; do [ "$a" = --report ] && REPORT=1 || RUN_ID=$a; done

echo "== VMs"
"${G[@]}" compute tpus tpu-vm list --zone "$ZONE" --format="table(name.basename(),acceptedType:label=TYPE,state,health)" 2>/dev/null \
  || "${G[@]}" compute tpus tpu-vm list --zone "$ZONE" --format="table(name.basename(),state)"
if "${G[@]}" storage ls "$RUN_GCS/single_vm/single_vm.log" > /dev/null 2>&1; then
  echo "== single-VM orchestrator (last 12 lines)"
  "${G[@]}" storage cat "$RUN_GCS/single_vm/single_vm.log" | tail -n 12
fi
if [ -z "$RUN_ID" ]; then
  RUN_ID=$("${G[@]}" storage ls "$RUN_GCS/" 2>/dev/null | sed 's#/$##; s#.*/##' | grep -v '^single_vm$' | sort | tail -1)
fi
[ -n "$RUN_ID" ] || { echo "no runs under $RUN_GCS yet"; exit 0; }
echo "== run $RUN_ID"
"${G[@]}" storage cat "$RUN_GCS/$RUN_ID/status.json" 2>/dev/null || echo "(no status.json yet)"
echo
echo "== watcher (last 15 lines)"
"${G[@]}" storage cat "$RUN_GCS/$RUN_ID/watch.log" 2>/dev/null | tail -n 15
if "${G[@]}" storage ls "$RUN_GCS/$RUN_ID/error_excerpt.txt" > /dev/null 2>&1; then
  echo "== error excerpt"; "${G[@]}" storage cat "$RUN_GCS/$RUN_ID/error_excerpt.txt" | tail -n 25
fi
if [ "$REPORT" = 1 ]; then
  dst="$ROOT/logs/fetched/$RUN_ID"; mkdir -p "$dst"
  for f in report.html final_results.json metrics.jsonl watch.log; do
    "${G[@]}" storage cp -q "$RUN_GCS/$RUN_ID/$f" "$dst/" 2>/dev/null
  done
  ls "$dst"
fi

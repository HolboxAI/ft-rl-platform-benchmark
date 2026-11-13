#!/usr/bin/env bash
# From the laptop: upload the code, queue a spot v6e-8 (one request per zone), and watch the DS-1000 run.
#
#   bash skyrl_ds1000/queue_ds1000.sh upload     # code -> gs://tpu-rl-run-cholec/code/ds1000-*.tgz
#   bash skyrl_ds1000/queue_ds1000.sh run        # queue, watch, delete the VM when the run ends
#   bash skyrl_ds1000/queue_ds1000.sh watch ZONE # watch (and delete when done) a VM that already exists
#
# The first request to become ACTIVE wins; the others are deleted at once (no second v6e-8 billing). The VM
# starts the pipeline itself (startup_ds1000.sh); this script only reads GCS. On preemption: queue again
# (the run starts over). When the run ends - completed, crashed, stalled or the pipeline failed - the logs are
# already in GCS; they are copied to logs/<run_id>/ and the VM is deleted. The VM's service account cannot
# delete the VM, so with this script stopped (laptop off) a finished VM keeps running until it is deleted.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
G=${GCLOUD:-$HOME/Downloads/google-cloud-sdk/bin/gcloud}
P=(--project tpu-rl-run)
CODE=gs://tpu-rl-run-cholec/code
GCS=gs://tpu-rl-run-cholec/runs
NODE=ds1000-skyrl-trainer
SKYRL_TPU_SRC=${SKYRL_TPU_SRC:-$(dirname "$(dirname "$ROOT")")/SkyRL-tpu}
Q_ZONES=${Q_ZONES:-"us-east1-d europe-west4-a us-east5-b us-east5-a us-central1-a"}
ATTEMPTS=${ATTEMPTS:-6}
ev() { echo "$(TZ=Asia/Kolkata date +"%Y-%m-%d %H:%M IST") EVENT $*"; }

upload() {
  local tmp; tmp=$(mktemp -d)
  tar czf "$tmp/ds1000-tpu_rl_benchmark.tgz" -C "$(dirname "$ROOT")/.." \
    --exclude='*/logs' --exclude='*/.venv*' --exclude='*/__pycache__' --exclude='*/miles-src' \
    --exclude='*/SkyRL-upstream-pr' --exclude='*/check_results' --exclude='.git' --exclude='*/miles-tpu' \
    tpu_rl_benchmark
  tar czf "$tmp/ds1000-SkyRL-tpu.tgz" -C "$SKYRL_TPU_SRC" --exclude='.git' --exclude='.venv*' --exclude='__pycache__' .
  ls -la "$tmp"
  $G storage cp -q "$tmp"/ds1000-*.tgz "$CODE/" && ev "uploaded to $CODE/"
  rm -rf "$tmp"
}

qr() { echo "ds1000-qr-$1"; }
qr_state() { $G compute tpus queued-resources describe "$(qr "$1")" "${P[@]}" --zone "$1" --format='value(state.state)' 2>/dev/null; }
qr_delete() { $G compute tpus queued-resources delete "$(qr "$1")" "${P[@]}" --zone "$1" --force --quiet --async > /dev/null 2>&1; }
qr_delete_eventually() {  # a request that is PROVISIONING cannot be deleted (Google: code 9); retry in the background
  ( for i in $(seq 1 90); do st=$(qr_state "$1"); [ -z "$st" ] && exit 0
      [ "$st" != PROVISIONING ] && qr_delete "$1"; sleep 20; done ) &
}
node_state() { $G compute tpus tpu-vm describe $NODE "${P[@]}" --zone "$1" --format='value(state)' 2>/dev/null; }
qr_create() {
  $G compute tpus queued-resources create "$(qr "$1")" "${P[@]}" --zone "$1" --node-id $NODE \
    --accelerator-type v6e-8 --runtime-version v2-alpha-tpuv6e --spot --valid-until-duration 24h \
    --scopes https://www.googleapis.com/auth/cloud-platform --metadata-from-file startup-script="$HERE/startup_ds1000.sh" 2>&1 \
    | tail -1 | cut -c1-160
}

queue_all() {
  for z in $Q_ZONES; do
    # synchronous: the create below fails with "already exists" while a delete is still running
    [ -n "$(qr_state "$z")" ] && $G compute tpus queued-resources delete "$(qr "$z")" "${P[@]}" --zone "$z" --force --quiet > /dev/null 2>&1
    ev "queued in $z: $(qr_create "$z")"
  done
}

wait_active() {  # -> echo the winning zone
  while true; do
    for z in $Q_ZONES; do
      st=$(qr_state "$z")
      if [ "$st" = ACTIVE ] && [ "$(node_state "$z")" = READY ]; then echo "$z"; return 0; fi
      case "$st" in FAILED|SUSPENDED) ev "request in $z is $st; re-queueing it" >&2; qr_delete "$z"; sleep 20; qr_create "$z" > /dev/null ;; esac
    done
    sleep 60
  done
}

collect() {  # run id -> copy its GCS logs to logs/<run_id>/
  local run=$1 dst="$ROOT/logs/${1:-ds1000_single_vm}"
  mkdir -p "$dst"
  [ -n "$run" ] && $G storage cp -q -r "$GCS/$run/*" "$dst/" > /dev/null 2>&1
  $G storage cp -q -r "$GCS/ds1000_single_vm/*" "$dst/" > /dev/null 2>&1
  ev "logs copied to $dst"
}

watch_zone() {  # zone -> returns when the VM is gone (preempted); exits when the run or pipeline ends
  local Z=$1 last="" lasts="" run_id="" st sv r s
  while true; do
    sleep 300
    st=$(node_state "$Z")
    sv=$($G storage cat "$GCS/ds1000_single_vm/single_vm.log" 2>/dev/null | tail -1)
    [ -n "$sv" ] && [ "$sv" != "$last" ] && { ev "stage: $sv"; last=$sv; }
    r=$($G storage cat "$GCS/ds1000_single_vm/single_vm.log" 2>/dev/null | grep -o "ds1000-skyrl-tpu-[0-9-]*" | tail -1); [ -n "$r" ] && run_id=$r
    if [ -n "$run_id" ]; then
      s=$($G storage cat "$GCS/$run_id/status.json" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['state'],'step',d.get('step'),'reward',d.get('train_reward'),'eval',d.get('eval_before'),d.get('eval_latest'))" 2>/dev/null)
      [ -n "$s" ] && [ "$s" != "$lasts" ] && { ev "run $run_id: $s"; lasts=$s; }
      case "$s" in
        COMPLETED*|CRASHED*|STALLED*)
          ev "RUN_END $s"; sleep 120; collect "$run_id"; qr_delete "$Z"; ev "VM + request deleted"
          case "$s" in COMPLETED*) exit 0 ;; *) exit 3 ;; esac ;;
      esac
    fi
    if echo "$sv" | grep -q "FAILED at stage"; then
      ev "PIPELINE_FAILED $sv"; sleep 60; collect ""; qr_delete "$Z"; ev "VM + request deleted"; exit 3
    fi
    case "$st" in READY|CREATING|STARTING|RESTARTING|"") ;; *) ev "VM $st (preempted?); queueing again"; return 0 ;; esac
  done
}

run() {
  for attempt in $(seq 1 "$ATTEMPTS"); do
    ev "attempt $attempt: queueing spot v6e-8 in: $Q_ZONES"
    $G storage rm -q "$GCS/ds1000_single_vm/single_vm.log" > /dev/null 2>&1
    queue_all
    Z=$(wait_active)
    for z in $Q_ZONES; do [ "$z" != "$Z" ] && qr_delete_eventually "$z"; done
    ev "GRANTED v6e-8 in $Z (other requests being deleted); the VM starts the pipeline by itself"
    watch_zone "$Z"
    qr_delete "$Z"; sleep 60
  done
  ev "GAVE_UP after $ATTEMPTS attempts"; exit 4
}

case "${1:-}" in
  upload) upload ;;
  run) run ;;
  watch) watch_zone "${2:?zone}"; ev "VM gone; not re-queueing in watch mode (use: $0 run)"; exit 2 ;;
  *) echo "usage: $0 upload|run|watch <zone>"; exit 1 ;;
esac

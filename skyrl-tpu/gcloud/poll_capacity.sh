#!/usr/bin/env bash
# Poll for spot v6e (Trillium) capacity until both VMs exist, then stop. Runs on the laptop.
#
#   bash gcloud/poll_capacity.sh                 # v6e-8 trainer + v6e-1 rollout, same zone
#   MAX_HOURS=12 bash gcloud/poll_capacity.sh
#   ROLLOUT_ACCEL=none bash gcloud/poll_capacity.sh   # single-VM layout: the v6e-8 only
#
# Round-robins the zones that offer v6e. In each zone: try the trainer (v6e-8); if it is
# granted, try the rollout VM (v6e-1) in the same zone until it is granted too.
# After every create call it checks what really exists, because a create can report an
# error while the VM was in fact created (seen in europe-west4-a on 2026-10-23).
# VMs in a FAILED / PREEMPTED / TERMINATED state are deleted so they cannot bill.
# Exits 0 with "GRANTED zone=<zone>" in the log once both VMs are READY; exits 3 after MAX_HOURS.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config/tpu.env"
ZONES=${ZONES:-"europe-west4-a us-east5-b us-east5-a us-east5-c us-central1-a us-south1-a us-east1-d"}
MAX_HOURS=${MAX_HOURS:-24}
ROUND_SLEEP=${ROUND_SLEEP:-120}
LOG="$ROOT/logs/capacity_poll.log"
mkdir -p "$ROOT/logs"
G=("$GCLOUD" --project "$PROJECT")
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

state_of() {  # name zone -> state or NONE
  "${G[@]}" compute tpus tpu-vm describe "$1" --zone "$2" --format='value(state)' 2>/dev/null || echo NONE
}

wait_settled() {  # name zone: wait (max 20 min) while the VM is still being created, so a pending
  # create is never abandoned for another zone (that could leave two v6e-8s billing)
  local i st
  for i in $(seq 1 40); do
    st=$(state_of "$1" "$2")
    case "$st" in CREATING|STARTING|RESTARTING) sleep 30 ;; *) return 0 ;; esac
  done
}

ensure() {  # name zone accel role -> 0 if READY
  local name=$1 zone=$2 accel=$3 role=$4 st
  st=$(state_of "$name" "$zone")
  case "$st" in
    READY) return 0 ;;
    CREATING|STARTING|RESTARTING) log "$name in $zone is $st; waiting"; wait_settled "$name" "$zone"; [ "$(state_of "$name" "$zone")" = READY ]; return ;;
    FAILED|PREEMPTED|TERMINATED|STOPPED|UNHEALTHY*)
      log "$name in $zone is $st; deleting it"; "${G[@]}" compute tpus tpu-vm delete "$name" --zone "$zone" --quiet > /dev/null 2>&1 ;;
  esac
  out=$("${G[@]}" compute tpus tpu-vm create "$name" --zone "$zone" --accelerator-type "$accel" \
        --version "$RUNTIME_VERSION" --spot --scopes https://www.googleapis.com/auth/cloud-platform \
        --labels "experiment=skyrl-cholect50,role=$role" 2>&1 | grep -vE '^\.+$' | tail -2 | tr '\n' ' ')
  wait_settled "$name" "$zone"
  st=$(state_of "$name" "$zone")
  log "$accel in $zone -> ${st} ${out:0:160}"
  [ "$st" = READY ]
}

log "polling spot v6e: trainer $TRAINER_ACCEL + rollout $ROLLOUT_ACCEL; zones: $ZONES; max ${MAX_HOURS}h"
round=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  round=$((round + 1)); log "round $round"
  for z in $ZONES; do
    if ensure "$TRAINER_VM" "$z" "$TRAINER_ACCEL" trainer; then
      [ "$ROLLOUT_ACCEL" = none ] && { log "GRANTED zone=$z (trainer READY; single-VM layout, no rollout VM)"; exit 0; }
      log "trainer READY in $z; now the rollout VM in the same zone"
      for i in $(seq 1 30); do
        ensure "$ROLLOUT_VM" "$z" "$ROLLOUT_ACCEL" rollout && { log "GRANTED zone=$z (trainer + rollout READY)"; exit 0; }
        [ "$(state_of "$TRAINER_VM" "$z")" = READY ] || { log "trainer lost in $z while waiting for rollout"; break; }
        sleep 60
      done
      log "no rollout capacity in $z after 30 tries; releasing the trainer to stop billing"
      "${G[@]}" compute tpus tpu-vm delete "$TRAINER_VM" --zone "$z" --quiet > /dev/null 2>&1
    fi
  done
  sleep "$ROUND_SLEEP"
done
log "gave up after ${MAX_HOURS}h"
exit 3

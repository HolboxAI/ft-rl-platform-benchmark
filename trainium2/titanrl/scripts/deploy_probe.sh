#!/bin/bash
# Deploy the TitanRL compatibility probe to the Trainium spot instance.
# Reads instance_titanrl.txt, waits for SSH, scp the probe script, launch it
# detached, print the remote log path.
set -uo pipefail

KEY="$HOME/.ssh/trainium-ap-south-2"
BASE=/home/susmitkulkarni/Documents/stanford_testing/cholec_launch
INST=$(cat "$BASE/instance_titanrl.txt")
IP=$(printf '%s\n' "$INST" | sed -n 's/^public_ip=//p' | tr -d '[:space:]')
[ -n "$IP" ] || { echo "NO IP in instance_titanrl.txt"; exit 1; }
HOST="ubuntu@$IP"
SSH="ssh -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
SCP="scp -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=30"

log() { echo "[deploy $(date '+%H:%M:%S')] $*"; }

log "waiting for SSH on $HOST"
READY=0
for i in $(seq 1 60); do
  if $SSH "$HOST" 'echo ok' 2>/dev/null | grep -q ok; then READY=1; log "SSH ready (attempt $i)"; break; fi
  sleep 10
done
[ "$READY" = "1" ] || { log "SSH NEVER READY"; exit 1; }

for f in setup_titanrl.sh; do
  ok=0
  for i in 1 2 3 4 5 6 7 8; do
    if $SCP "$BASE/$f" "$HOST:~/$f" 2>/dev/null; then ok=1; log "scp $f OK"; break; fi
    sleep 3
  done
  [ "$ok" = "1" ] || { log "SCP $f FAILED"; exit 1; }
done

log "launching setup_titanrl.sh detached"
$SSH "$HOST" 'rm -f ~/titanrl_probe.log; setsid bash -lc "bash ~/setup_titanrl.sh > ~/titanrl_probe.log 2>&1" </dev/null >/dev/null 2>&1 &' 2>/dev/null
log "DEPLOY COMPLETE. remote log: ~/titanrl_probe.log"

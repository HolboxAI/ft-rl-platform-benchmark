#!/bin/bash
# Poll the TitanRL probe to a terminal state, pull the log home, and terminate
# the spot instance.
set -uo pipefail

KEY="$HOME/.ssh/trainium-ap-south-2"
BASE=/home/susmitkulkarni/Documents/stanford_testing/cholec_launch
INST=$(cat "$BASE/instance_titanrl.txt")
IP=$(printf '%s\n' "$INST" | sed -n 's/^public_ip=//p' | tr -d '[:space:]')
IID=$(printf '%s\n' "$INST" | sed -n 's/^instance_id=//p' | tr -d '[:space:]')
REGION=ap-south-2
HOST="ubuntu@$IP"
SSH="ssh -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
SCP="scp -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no"
MAX_ITER="${1:-40}"      # 40 * 30s = ~20 min
TERMINATE=1
[ "${2:-}" = "--no-terminate" ] && TERMINATE=0

log() { echo "[poll $(date '+%H:%M:%S')] $*"; }

stop_instance() {
  if [ "$TERMINATE" = "1" ] && [ -n "$IID" ]; then
    aws ec2 terminate-instances --instance-ids "$IID" --region "$REGION" >/dev/null 2>&1
    log "TERMINATE sent for $IID"
  else
    log "terminate skipped"
  fi
}

for i in $(seq 1 "$MAX_ITER"); do
  SNAP=$($SSH "$HOST" "
    if pgrep -f '[s]etup_titanrl.sh' >/dev/null 2>&1; then A=1; else A=0; fi
    B=\$(stat -c%s ~/titanrl_probe.log 2>/dev/null || echo 0)
    D=\$(grep -c 'TITANRL_PROBE_COMPLETE' ~/titanrl_probe.log 2>/dev/null || echo 0)
    E=\$(tail -c 4000 ~/titanrl_probe.log 2>/dev/null | tr '\r' '\n' | grep -aoE 'Traceback|RuntimeError|ModuleNotFoundError|ImportError|error:|ERROR:|FAILED:' | tail -1)
    echo \"ALIVE=\$A\"
    echo \"BYTES=\$B\"
    echo \"DONE=\$D\"
    echo \"ERR=\$E\"
  " 2>/dev/null)

  ALIVE=$(echo "$SNAP" | sed -n 's/^ALIVE=//p')
  BYTES=$(echo "$SNAP" | sed -n 's/^BYTES=//p')
  DONE=$(echo "$SNAP" | sed -n 's/^DONE=//p')
  ERR=$(echo "$SNAP" | sed -n 's/^ERR=//p')

  if [ "$DONE" = "1" ] || [ "$DONE" -ge 1 ] 2>/dev/null; then
    $SCP "$HOST:~/titanrl_probe.log" "$BASE/titanrl_probe.log" >/dev/null 2>&1
    log "SUCCESS — TITANRL_PROBE_COMPLETE; terminating instance"
    stop_instance
    exit 0
  fi

  if [ "$ALIVE" != "1" ] && [ "$BYTES" -gt 0 ]; then
    $SCP "$HOST:~/titanrl_probe.log" "$BASE/titanrl_probe.log" >/dev/null 2>&1
    log "PROCESS_ENDED (no COMPLETE marker) — terminating; check log"
    $SSH "$HOST" "tail -40 ~/titanrl_probe.log" 2>/dev/null
    stop_instance
    exit 0
  fi

  if [ $((i % 4)) -eq 0 ]; then
    log "ok iter $i: alive=$ALIVE bytes=$BYTES err=$ERR"
  fi
  sleep 30
done

log "TIMEOUT after $MAX_ITER iters"
$SSH "$HOST" "tail -40 ~/titanrl_probe.log" 2>/dev/null

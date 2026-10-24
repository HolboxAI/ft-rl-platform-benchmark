#!/bin/bash
# Regular healthcheck for the Trainium2 GRPO run.
#
# Polls the remote spot instance and, on any terminal state (success, crash,
# stall, or process death), first copies the run log home so the result/error is
# never lost, then TERMINATES the instance to stop the per-second spot billing.
# Run every few minutes from cron; it is idempotent (terminate is a no-op if the
# instance is already gone).
#
# Usage: healthcheck.sh [--no-terminate]
#   --no-terminate   kill the process but leave the instance running (debug only)
set -u

KEY="/home/susmitkulkarni/.ssh/trainium-ap-south-2"
HOST="ubuntu@18.60.144.22"
SSH="/usr/bin/ssh -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=15"
SCP="/usr/bin/scp -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no"
INSTANCE_ID="i-0d3797ee48e3d9664"
REGION="ap-south-2"
BASE="/home/susmitkulkarni/Documents/stanford_testing/ft-rl-platform-benchmark/trainium2/logs"
LOG="$BASE/healthcheck.log"
STATE="$BASE/healthcheck.state"   # per-log: "<path> <bytes> <stalled_since_epoch|0>"
STALL_SECONDS=1200                 # alive but log unchanged this long -> kill
TERMINATE=1
[ "${1:-}" = "--no-terminate" ] && TERMINATE=0
mkdir -p "$BASE"

ts() { date '+%Y-%m-%dT%H:%M:%S'; }
note() { echo "$(ts) $*" >> "$LOG"; }

# Terminate the spot instance. Logs the result but never fails the script.
stop_instance() {
  if [ "$TERMINATE" = "1" ]; then
    /usr/bin/aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
      >> "$LOG" 2>&1
    note "TERMINATE sent for $INSTANCE_ID"
  else
    note "terminate skipped (--no-terminate)"
  fi
}

# Pull the run log home before we tear the instance down.
fetch_log() {
  local remote="$1"
  local name; name="$(basename "$remote")"
  $SCP "$HOST:$remote" "$BASE/$name" >> "$LOG" 2>&1 && \
    note "log fetched -> $BASE/$name"
}

# --- remote snapshot (single SSH round-trip) ---
REMOTE_LOG=$($SSH "$HOST" 'ls -t ~/run_grpo_*.log 2>/dev/null | head -1' 2>/dev/null)
if [ -z "$REMOTE_LOG" ]; then
  note "NO_LOG: no run_grpo_*.log on instance (instance gone or run never started)"
  exit 0
fi

# Each field is emitted as a labeled KEY=VALUE line so an EMPTY value (e.g. no
# error yet) still occupies a line. Positional sed on a 4-line blob is fragile:
# when the error grep yields nothing it contributes no line at all, so the byte
# count shifts up into the "error" slot (the false-positive that killed a healthy
# run once). Labels make the parse immune to that.
SNAP=$($SSH "$HOST" "
  if pgrep -f '^python grpo_train' >/dev/null 2>&1; then A=1; else A=0; fi
  S=\$(grep -aoE '[0-9]+/64' $REMOTE_LOG 2>/dev/null | tail -1)
  E=\$(tail -c 3000 $REMOTE_LOG 2>/dev/null | tr '\r' '\n' | grep -aoE 'NRT EXECUTION FAILED|Failed to allocate resource|Traceback|RuntimeError|EVAL_RESULT' | tail -1)
  B=\$(stat -c%s $REMOTE_LOG 2>/dev/null)
  echo \"ALIVE=\$A\"
  echo \"STEP=\$S\"
  echo \"ERR=\$E\"
  echo \"BYTES=\$B\"
" 2>/dev/null)

if [ -z "$SNAP" ]; then
  note "SSH_FAIL: cannot reach instance (already terminated or transient) — no action"
  exit 0
fi

ALIVE=$(echo "$SNAP" | sed -n 's/^ALIVE=//p')
STEP=$(echo "$SNAP" | sed -n 's/^STEP=//p')
ERR=$(echo "$SNAP" | sed -n 's/^ERR=//p')
BYTES=$(echo "$SNAP" | sed -n 's/^BYTES=//p')
STEP="${STEP:-?}"

# --- terminal-state decisions (each preserves the log, then stops billing) ---
if [ "$ERR" = "EVAL_RESULT" ]; then
  fetch_log "$REMOTE_LOG"
  note "SUCCESS $STEP ($REMOTE_LOG) — eval complete, terminating instance"
  stop_instance
  exit 0
fi

if [ -n "$ERR" ]; then
  fetch_log "$REMOTE_LOG"
  note "FAILED $STEP err=$ERR ($REMOTE_LOG) — killing process + terminating instance"
  $SSH "$HOST" "pkill -f '^python grpo_train'" 2>/dev/null
  stop_instance
  exit 0
fi

if [ "$ALIVE" != "1" ]; then
  fetch_log "$REMOTE_LOG"
  note "PROCESS_DEAD $STEP ($REMOTE_LOG) — terminating instance"
  stop_instance
  exit 0
fi

# --- alive: stall detection via a persistent state file ---
touch "$STATE"
PREV_BYTES=""
PREV_STALL=""
line=$(grep -F "$REMOTE_LOG " "$STATE" | tail -1)
if [ -n "$line" ]; then
  PREV_BYTES=$(echo "$line" | awk '{print $2}')
  PREV_STALL=$(echo "$line" | awk '{print $3}')
fi

NOW=$(date +%s)
if [ -n "$PREV_BYTES" ] && [ "$BYTES" -le "$PREV_BYTES" ]; then
  # log not advancing (or shrank). Mark the stall start if not already marked.
  if [ -z "$PREV_STALL" ] || [ "$PREV_STALL" = "0" ]; then
    STALL="$NOW"
    note "STALL_START $STEP ${BYTES}B (unchanged from ${PREV_BYTES}B)"
  else
    STALL="$PREV_STALL"
    if [ $((NOW - STALL)) -ge "$STALL_SECONDS" ]; then
      fetch_log "$REMOTE_LOG"
      note "STALLED $STEP ${BYTES}B for $(( (NOW-STALL)/60 ))min — killing + terminating instance"
      $SSH "$HOST" "pkill -f '^python grpo_train'" 2>/dev/null
      stop_instance
      exit 0
    fi
  fi
else
  STALL="0"
fi
# rewrite this log's state line (drop the old one)
grep -vF "$REMOTE_LOG " "$STATE" > "$STATE.tmp" 2>/dev/null || true
echo "$REMOTE_LOG $BYTES $STALL" >> "$STATE.tmp"
mv "$STATE.tmp" "$STATE"
note "OK $STEP ${BYTES}B (alive, progressing)"

#!/bin/bash
# Poll the remote GRPO log until it reaches a terminal state (crash or a
# completed step / eval), then print the relevant tail. Bounded by MAX_ITER.
set -u
KEY="$HOME/.ssh/trainium-ap-south-2"
HOST="ubuntu@16.113.78.135"
SSH="ssh -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no"
LOG="~/run_grpo.log"
MAX_ITER="${1:-40}"   # ~40 * 15s = 10 min max

for i in $(seq 1 "$MAX_ITER"); do
  out=$($SSH "$HOST" "cat $LOG 2>/dev/null" 2>/dev/null)
  if echo "$out" | grep -qE "EVAL_RESULT"; then
    echo "=== DONE (iter $i) ==="
    echo "$out" | grep -E "EVAL_RESULT" | tail -5
    exit 0
  fi
  if echo "$out" | grep -qE "Failed to allocate DEVICE memory|OOM recovery failed|Traceback|Error: |error: |ValueError|TypeError|AttributeError|RuntimeError"; then
    echo "=== TERMINAL/STUCK (iter $i) ==="
    echo "$out" | tail -60
    exit 0
  fi
  # print a heartbeat line every 4 iterations
  if [ $((i % 4)) -eq 0 ]; then
    echo "--- poll $i: last line: $(echo "$out" | grep -v '^$' | tail -1)"
  fi
  sleep 15
done
echo "=== TIMEOUT: last 80 lines ==="
$SSH "$HOST" "tail -80 $LOG 2>/dev/null" 2>/dev/null

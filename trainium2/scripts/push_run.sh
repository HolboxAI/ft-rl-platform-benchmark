#!/bin/bash
# Push a script to the Trainium instance and run it detached under the
# TorchNeuron venv, logging to a remote file. Robust to banner-exchange
# timeouts: each ssh/scp is retried.
set -u
KEY="$HOME/.ssh/trainium-ap-south-2"
HOST="ubuntu@16.113.78.135"
LOCAL="$1"
REMOTE="~/$(basename "$1")"
LOG="~/run_$(basename "$1").log"
SSH="ssh -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=15"
SCP="scp -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=15"

scp_done=0
for i in 1 2 3 4 5 6 7 8; do
  if $SCP "$LOCAL" "$HOST:$REMOTE" 2>/dev/null; then scp_done=1; echo "scp OK (attempt $i)"; break; fi
  echo "scp retry $i"; sleep 3
done
[ "$scp_done" = "1" ] || { echo "SCP FAILED"; exit 1; }

run_done=0
for i in 1 2 3 4 5 6 7 8; do
  if $SSH "$HOST" "setsid bash -lc 'source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate && cd ~ && python $REMOTE > $LOG 2>&1' </dev/null &" 2>/dev/null; then
    echo "LAUNCHED (attempt $i)"; run_done=1; break
  fi
  echo "run retry $i"; sleep 3
done
[ "$run_done" = "1" ] || { echo "RUN LAUNCH FAILED"; exit 1; }
echo "remote log: $LOG"

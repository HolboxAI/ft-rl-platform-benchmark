#!/bin/bash
# Push grpo_train.py + reward_tr.py and launch the GRPO smoke test detached.
set -u
KEY="$HOME/.ssh/trainium-ap-south-2"
HOST="ubuntu@16.113.78.135"
SSH="ssh -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=15"
SCP="scp -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=15"
DIR="/home/susmitkulkarni/Documents/stanford_testing/ft-rl-platform-benchmark/trainium2/scripts"
LOG="~/run_grpo.log"

for f in grpo_train.py reward_tr.py; do
  ok=0
  for i in 1 2 3 4 5 6 7 8; do
    if $SCP "$DIR/$f" "$HOST:~/$f" 2>/dev/null; then ok=1; echo "scp $f OK (attempt $i)"; break; fi
    echo "scp $f retry $i"; sleep 3
  done
  [ "$ok" = "1" ] || { echo "SCP $f FAILED"; exit 1; }
done

run_env="SMOKE=1 MAX_STEPS=4 TORCH_COMPILE=0 NEURON_RT_VISIBLE_CORES=0 BETA=0.0 USE_LORA=1 LORA_R=16 LORA_ALPHA=32 NKI_ENABLE_TRACE_CACHE=0 BATCH=1 NUM_GEN=8 GEN_BATCH=8"
for i in 1 2 3 4 5 6 7 8; do
  if $SSH "$HOST" "setsid bash -lc 'source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate && cd ~ && rm -f $LOG && $run_env python grpo_train.py > $LOG 2>&1' </dev/null &" 2>/dev/null; then
    echo "LAUNCHED (attempt $i) with: $run_env"; break
  fi
  echo "launch retry $i"; sleep 3
done
echo "remote log: $LOG"

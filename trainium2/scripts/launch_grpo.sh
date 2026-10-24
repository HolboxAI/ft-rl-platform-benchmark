#!/bin/bash
# Push grpo_train.py + reward_tr.py and launch the GRPO smoke test detached.
set -u
KEY="$HOME/.ssh/trainium-ap-south-2"
HOST="ubuntu@18.60.144.22"
SSH="ssh -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=15"
SCP="scp -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o ServerAliveInterval=15"
DIR="/home/susmitkulkarni/Documents/stanford_testing/ft-rl-platform-benchmark/trainium2/scripts"
# Unique log per launch. Never `rm -f` a shared log inside the retry loop: a
# banner-timeout "failure" on attempt 1 can still have started the remote
# process, and the retry's `rm -f` then unlinks the live log and spawns a
# duplicate that dies with "Failed to initialize Neuron Runtime" (device held).
LOG="~/run_grpo_$(date +%s).log"

for f in grpo_train.py reward_tr.py; do
  ok=0
  for i in 1 2 3 4 5 6 7 8; do
    if $SCP "$DIR/$f" "$HOST:~/$f" 2>/dev/null; then ok=1; echo "scp $f OK (attempt $i)"; break; fi
    echo "scp $f retry $i"; sleep 3
  done
  [ "$ok" = "1" ] || { echo "SCP $f FAILED"; exit 1; }
done

# Full 1-epoch run: all 256 train / 32 val rows, batch 4, n=8 -> 64 steps.
# Batch 4 (not 16): batch 16 exceeds the 300k-instruction compiler limit
# (NCC_EXTP003), and 64 steps is exactly one pass over 256 rows.
# GEN_BATCH must be a multiple of BOTH num_generations (8) and per_device_batch
# (4), so 8 works (= 2 prompts/call).
# (Smoke config for reference: SMOKE=1 MAX_STEPS=4 BATCH=1 NUM_GEN=8 GEN_BATCH=8)
run_env="SMOKE=0 MAX_STEPS=64 TORCH_COMPILE=0 NEURON_RT_VISIBLE_CORES=0 BETA=0.0 USE_LORA=1 LORA_R=16 LORA_ALPHA=32 NKI_ENABLE_TRACE_CACHE=0 BATCH=4 NUM_GEN=8 GEN_BATCH=8"
# Idempotent launch: refuse to start if a grpo_train process already holds the
# single NeuronCore (a duplicate can never init the runtime and would only
# pollute logs). The guard lives inside the remote command so it is re-checked
# on every retry, not just once.
for i in 1 2 3 4 5 6 7 8; do
  if $SSH "$HOST" "setsid bash -lc 'source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate && cd ~ && if pgrep -f \"^python grpo_train\" >/dev/null 2>&1; then echo \"ALREADY RUNNING: skipped duplicate launch\" >> ~/launch_guard.log; else $run_env python grpo_train.py > $LOG 2>&1; fi' </dev/null &" 2>/dev/null; then
    echo "LAUNCHED (attempt $i) with: $run_env"; break
  fi
  echo "launch retry $i"; sleep 3
done
echo "remote log: $LOG"

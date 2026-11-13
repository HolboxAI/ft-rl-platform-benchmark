#!/usr/bin/env bash
# Single-VM layout: setup, rollout server, probes and the run, all on ONE v6e-8. Runs on that VM.
#
#   nohup bash ~/tpu_rl_benchmark/skyrl-tpu/ops/single_vm.sh > ~/tpu_rl_benchmark/skyrl-tpu/logs/single_vm.out 2>&1 &
#   (gcloud/vm.sh single-start does exactly this from the laptop)
#
# Chips: policy 0-3 (TorchTPU, FSDP), ref 4 (1 chip), rollout server 6 (vLLM-TPU, or HF with ENGINE=hf).
# Weights reach the rollout server through /dev/shm (reload backend, `path` transport).
# Every stage is logged to logs/single_vm.log and mirrored to $RUN_GCS/single_vm/, so progress is
# visible with the laptop off (gcloud/run_status.sh shows it). Stops at the first failed stage.
#
# Env: ENGINE=hf|vllm (default hf: vLLM-TPU 0.32 ran its experimental Qwen vision tower at ~80 s per frame,
#      2026-10-24; with ENGINE=vllm it falls back to hf if the vLLM server does not come up when
#      ENGINE_FALLBACK=1, the default), SKIP_PROBES=1, RUN_ARGS="trainer.max_training_steps=1" (smoke).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TOPOLOGY=single
ENGINE=${ENGINE:-hf}
ENGINE_FALLBACK=${ENGINE_FALLBACK:-1}
ROLLOUT_CHIPS=${ROLLOUT_CHIPS:-6}
RUN_GCS=${RUN_GCS:-gs://tpu-rl-run-cholec/runs}
DATA_GCS=${DATA_GCS:-gs://tpu-rl-run-cholec/cholec}
MODEL=${MODEL:-$HOME/models/cholec-sft-qwen3.5-4b}
LOG="$ROOT/logs/single_vm.log"
mkdir -p "$ROOT/logs"
export PATH="$HOME/.local/bin:$PATH"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; gcloud storage cp -q "$LOG" "$RUN_GCS/single_vm/single_vm.log" > /dev/null 2>&1; }
fail() { log "FAILED at stage: $1"; for f in "$ROOT"/logs/*.log; do gcloud storage cp -q "$f" "$RUN_GCS/single_vm/" > /dev/null 2>&1; done; exit 1; }
stage() { log "== $1"; }

stage "setup: trainer venv"
if [ -x "$HOME/.venv-skyrl-tpu/bin/python" ] && "$HOME/.venv-skyrl-tpu/bin/python" -c "import torch_tpu, skyrl" 2>/dev/null; then
  log "trainer venv present"
else
  bash "$ROOT/setup/setup_trainer_vm.sh" > "$ROOT/logs/setup_trainer.log" 2>&1 || fail "setup_trainer_vm.sh (logs/setup_trainer.log)"
fi

stage "setup: rollout venv(s)"
for e in "$ENGINE" $( [ "$ENGINE_FALLBACK" = 1 ] && [ "$ENGINE" = vllm ] && echo hf ); do
  if [ -x "$HOME/.venv-rollout-$e/bin/python" ]; then log "rollout venv $e present"; continue; fi
  ENGINE=$e bash "$ROOT/setup/setup_rollout_vm.sh" > "$ROOT/logs/setup_rollout_$e.log" 2>&1 \
    || { [ "$e" = "$ENGINE" ] && [ "$ENGINE_FALLBACK" != 1 ] && fail "setup_rollout_vm.sh ENGINE=$e"; log "rollout venv $e setup FAILED (logs/setup_rollout_$e.log)"; }
done

stage "data"
if [ -f "$HOME/cholec/skyrl_parquet/val.parquet" ] && [ -f "$MODEL/config.json" ]; then
  log "data present"
else
  DATA_GCS=$DATA_GCS bash "$ROOT/setup/fetch_data.sh" > "$ROOT/logs/fetch_data.log" 2>&1 || fail "fetch_data.sh"
fi

start_rollout() {  # engine -> 0 when healthy
  local e=$1
  pkill -f rollout_supervisor.py 2>/dev/null; pkill -f rollout_server.py 2>/dev/null; sleep 5
  ENGINE=$e ROLLOUT_CHIPS=$ROLLOUT_CHIPS MODEL=$MODEL WORK_DIR=/dev/shm/rollout_weights DETACH=1 \
    bash "$ROOT/rollout/start_rollout.sh" >> "$LOG" 2>&1
  for i in $(seq 1 180); do  # first TPU compile can take a while
    curl -sf -m 5 http://localhost:8000/health > /dev/null && return 0
    pgrep -f rollout_supervisor.py > /dev/null || return 1
    sleep 10
  done
  return 1
}

stage "rollout server on chip $ROLLOUT_CHIPS (engine $ENGINE)"
running_engine=$(curl -sf -m 5 http://localhost:8000/version 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('engine',''))" 2>/dev/null)
if [ -n "$running_engine" ]; then
  # A rerun after a later stage failed: keep the healthy server instead of recompiling it.
  ENGINE=$running_engine
  log "rollout server already healthy (engine $ENGINE); reusing it"
elif start_rollout "$ENGINE"; then
  log "rollout server healthy: $(curl -s -m 5 http://localhost:8000/version)"
elif [ "$ENGINE" = vllm ] && [ "$ENGINE_FALLBACK" = 1 ] && [ -x "$HOME/.venv-rollout-hf/bin/python" ]; then
  log "vLLM-TPU server did not come up (logs/rollout/); falling back to the HF engine"
  ENGINE=hf
  start_rollout hf || fail "rollout server (vllm and hf)"
  log "rollout server healthy: $(curl -s -m 5 http://localhost:8000/version)"
else
  fail "rollout server ($ENGINE)"
fi
export ENGINE

# shellcheck disable=SC1091
source "$ROOT/skyrl_cholect50/tpu_env.sh"

if [ "${SKIP_PROBES:-0}" != 1 ]; then
  stage "probe P0: TorchTPU policy (chips $POLICY_CHIPS) + ref (chips $REF_CHIPS) next to the rollout server"
  (cd "$SKYRL_TPU_DIR" && python "$ROOT/probes/p0_torchtpu.py" --timeout 1800) > "$ROOT/logs/p0.log" 2>&1 \
    || { gcloud storage cp -q "$ROOT/logs/p0_torchtpu.json" "$RUN_GCS/single_vm/" 2>/dev/null; fail "P0 (logs/p0.log)"; }
  log "P0 passed"
  stage "probe P1: rollout server on 32 real val frames + reload"
  (cd "$SKYRL_TPU_DIR" && python "$ROOT/probes/p1_rollout.py" --url http://localhost:8000 --n 32 --reload \
     --model "$MODEL" --parquet "$HOME/cholec/skyrl_parquet/val.parquet") > "$ROOT/logs/p1.log" 2>&1 \
    || { gcloud storage cp -q "$ROOT/logs/p1_rollout.json" "$RUN_GCS/single_vm/" 2>/dev/null; fail "P1 (logs/p1.log)"; }
  log "P1 passed: $(python -c "import json;d=json.load(open('$ROOT/logs/p1_rollout.json'));print(d['stages'].get('eval_greedy',{}).get('mean_reward'))") greedy val reward (GPU run: 0.329)"
  for f in p0_torchtpu.json p1_rollout.json; do gcloud storage cp -q "$ROOT/logs/$f" "$RUN_GCS/single_vm/" 2>/dev/null; done
fi

stage "launch run + watcher (TOPOLOGY=single, engine $ENGINE)"
# shellcheck disable=SC2086
bash "$ROOT/ops/launch_bg.sh" ${RUN_ARGS:-} >> "$LOG" 2>&1 || fail "launch_bg.sh"
log "run launched: $(cat "$ROOT/logs/CURRENT_RUN"); the watcher takes over from here"

#!/usr/bin/env bash
# TRL CholecT50 GRPO on Trainium (native TorchNeuron): the job the per-boot runner executes.
# Environment comes from control/job.env (written by submit.sh). Stages (STAGES):
#   env      activate the Neuron venv; record torch / torch-neuronx / neuron-ls
#   install  trl + accelerate + transformers + peft on top, torch stack frozen by constraints
#   data     SFT checkpoint + CholecT50 parquet from S3 (hash-checked)
#   check    hf-trl/scripts/check_env.py: neuron device, accelerate device, stack, model + 16-token generate
#   smoke    1 GRPO step, generation in chunks of 16 completions (2 frames x 8)
#   full     16 GRPO steps + greedy val eval before/after
set -uo pipefail
: "${S3_ROOT:?}" "${RUN_ID:?}" "${MODEL_S3_URI:?}" "${PARQUET_S3_URI:?}"
STAGES=${STAGES:-"env install data check smoke"}
DATA=/data
CODE=$DATA/code/rl-benchmark
OUT=$DATA/runs/$RUN_ID
MODEL=$DATA/models/cholec-sft-qwen3.5-4b
mkdir -p "$OUT" /data/state "$DATA/models"
export HOME=${HOME:-/root} PYTHONUNBUFFERED=1 HF_HOME=$DATA/hf TOKENIZERS_PARALLELISM=false
export NEURON_COMPILE_CACHE_URL=$DATA/neuron-compile-cache
log() { echo "[$(date -u +%FT%TZ)] $*"; }
stage() { echo "$1" > /data/state/current_stage; log "=== stage $1"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
sync_up() { aws s3 sync --only-show-errors "$OUT/" "$S3_ROOT/runs/$RUN_ID/out/" --exclude "*/trainer/*" || true; }
trap sync_up EXIT
( while sleep 300; do sync_up; done ) &

aws s3 sync --delete --only-show-errors "$S3_ROOT/code/" "$CODE/"
log "job $RUN_ID stages: $STAGES"

# ---- env (always: every later stage needs the venv)
stage env
VENV=$(ls -d /opt/aws_neuronx_venv_torch_neuronx_* /opt/aws_neuronx_venv_pytorch* 2>/dev/null | sort -V | tail -1)
[ -n "$VENV" ] || { log "FATAL: no Neuron PyTorch venv under /opt"; exit 2; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PATH="/opt/aws/neuron/bin:$PATH"
{
  echo "# $(date -u +%FT%TZ) venv=$VENV"; python --version; neuron-ls 2>&1 || true
  pip list 2>/dev/null | grep -iE "^(torch|torch-neuronx|neuronx-cc|torchvision|transformers|trl|peft|accelerate|datasets) "
} | tee "$OUT/environment.txt"
pip freeze 2>/dev/null | grep -iE "^(torch|torch-neuronx|neuronx-cc|libneuronxla)==" > "$OUT/constraints.txt"

if want install; then
  stage install
  pip install -c "$OUT/constraints.txt" "trl==${TRL_VERSION:-1.16.0}" "accelerate==${ACCELERATE_VERSION:-1.15.0}" \
    "transformers==${TRANSFORMERS_VERSION:-5.19.0}" peft datasets pandas pyarrow pillow "torchvision" 2>&1 | tail -4 \
    | tee "$OUT/install.txt"
  pip list 2>/dev/null | grep -iE "^(torch|torch-neuronx|torchvision|transformers|trl|peft|accelerate|datasets) " \
    | tee -a "$OUT/environment.txt"
  python -c "import torch; assert torch.__version__.startswith('$(grep -i '^torch==' "$OUT/constraints.txt" | cut -d= -f3)'), torch.__version__" \
    || { log "FATAL: install replaced the Neuron torch build"; exit 2; }
fi

if want data; then
  stage data
  aws s3 sync --only-show-errors "$MODEL_S3_URI/" "$MODEL/"
  mkdir -p "$CODE/data/cholect50/parquet"
  aws s3 sync --only-show-errors "$PARQUET_S3_URI/" "$CODE/data/cholect50/parquet/"
  (cd "$CODE" && sha256sum data/cholect50/parquet/train.parquet data/cholect50/parquet/val.parquet) | tee "$OUT/data_manifest.txt"
fi
[ -f "$CODE/data/cholect50/parquet/train.parquet" ] || aws s3 sync --only-show-errors "$PARQUET_S3_URI/" "$CODE/data/cholect50/parquet/"

if want check; then
  stage check
  python "$CODE/hf-trl/scripts/check_env.py" --model "$MODEL" --data-dir "$CODE/data/cholect50/parquet" \
    --out "$OUT/check_env.json" 2>&1 | tee "$OUT/check_console.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "check FAILED: see check_env.json"; exit 3; }
fi

run_train() {  # $1 = name, rest = extra args
  local name=$1; shift
  mkdir -p "$OUT/$name"
  accelerate launch --num_processes "${NUM_PROCESSES:-1}" "$CODE/hf-trl/scripts/train_grpo_trl.py" \
    --model "$MODEL" --data-dir "$CODE/data/cholect50/parquet" --out-dir "$OUT/$name" \
    --hourly-usd "${PRICE_PER_HOUR_USD:-0}" --platform "hf-trl-${INSTANCE_TYPE:-trainium}" "$@" \
    2>&1 | tee "$OUT/$name/console.log"
  return "${PIPESTATUS[0]}"
}

if want smoke; then
  stage smoke
  t0=$(date +%s)
  run_train smoke --max-steps 1 --steps-per-generation 16 --skip-eval \
    || { log "smoke FAILED after $(( $(date +%s) - t0 )) s"; exit 4; }
  log "smoke done in $(( $(date +%s) - t0 )) s"
fi

if want full; then
  stage full
  t0=$(date +%s)
  run_train full --steps-per-generation "${STEPS_PER_GENERATION:-16}" \
    || { log "full FAILED after $(( $(date +%s) - t0 )) s"; exit 5; }
  log "full done in $(( $(date +%s) - t0 )) s"
fi

stage complete
log "job $RUN_ID complete"

#!/usr/bin/env bash
# Our plain-PyTorch GRPO loop (common/grpo) on Trainium, native TorchNeuron: the job the
# per-boot runner executes. Environment comes from control/job.env (written by submit.sh).
#   env      activate the Neuron venv; record torch / torch-neuronx / neuron-ls
#   install  transformers + peft + data libs on top, torch stack frozen by constraints
#   data     SFT checkpoint + CholecT50 parquet from S3 (hash-checked)
#   smoke    common/smoke_test.py --backend neuron: device, load, forward x2, LoRA train step x2,
#            16-token static sampling + reward, at the run's shapes (2048-token prompt)
#   rl1      common/grpo/train_grpo.py, 1 GRPO step (16 frames x 8), no eval
#   full     16 steps + greedy val eval before/after
set -uo pipefail
: "${S3_ROOT:?}" "${RUN_ID:?}" "${MODEL_S3_URI:?}" "${PARQUET_S3_URI:?}"
STAGES=${STAGES:-"env install data smoke rl1"}
DATA=/data
CODE=$DATA/code/rl-benchmark
OUT=$DATA/runs/$RUN_ID
MODEL=$DATA/models/cholec-sft-qwen3.5-4b
mkdir -p "$OUT" /data/state "$DATA/models"
export HOME=${HOME:-/root} PYTHONUNBUFFERED=1 HF_HOME=$DATA/hf TOKENIZERS_PARALLELISM=false
export NEURON_COMPILE_CACHE_URL=$DATA/neuron-compile-cache NEURON_MODE=native
export INSTANCE_TYPE=${INSTANCE_TYPE:-trn1.2xlarge} HOURLY_USD=${PRICE_PER_HOUR_USD:-0}
log() { echo "[$(date -u +%FT%TZ)] $*"; }
stage() { echo "$1" > /data/state/current_stage; log "=== stage $1"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
sync_up() { aws s3 sync --only-show-errors "$OUT/" "$S3_ROOT/runs/$RUN_ID/out/" || true; }
trap sync_up EXIT
( while sleep 300; do sync_up; done ) &

aws s3 sync --delete --only-show-errors "$S3_ROOT/code/" "$CODE/"
log "job $RUN_ID stages: $STAGES"

stage env
VENV=$(ls -d /opt/aws_neuronx_venv_torch_neuronx_* /opt/aws_neuronx_venv_pytorch* 2>/dev/null | sort -V | tail -1)
[ -n "$VENV" ] || { log "FATAL: no Neuron PyTorch venv under /opt"; exit 2; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PATH="/opt/aws/neuron/bin:$PATH"
{
  echo "# $(date -u +%FT%TZ) venv=$VENV"; python --version; neuron-ls 2>&1 || true
  pip list 2>/dev/null | grep -iE "^(torch|torch-neuronx|neuronx-cc|torchvision|transformers|peft) "
} | tee "$OUT/environment.txt"
pip freeze 2>/dev/null | grep -iE "^(torch|torch-neuronx|neuronx-cc|libneuronxla)==" > "$OUT/constraints.txt"

if want install; then
  stage install
  pip install -c "$OUT/constraints.txt" "transformers==${TRANSFORMERS_VERSION:-5.19.0}" peft pandas pyarrow pillow \
    torchvision 2>&1 | tail -4 | tee "$OUT/install.txt"
  pip list 2>/dev/null | grep -iE "^(torch|torch-neuronx|torchvision|transformers|peft) " | tee -a "$OUT/environment.txt"
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
cd "$CODE"

if want smoke; then
  stage smoke
  python -m common.smoke_test --backend neuron --model "$MODEL" --data "$CODE/data/cholect50/parquet" \
    --out "$OUT/smoke.json" 2>&1 | tee "$OUT/smoke_console.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "smoke FAILED: see smoke.json"; exit 3; }
fi

run_grpo() {  # $1 = name, rest = extra args
  local name=$1; shift
  mkdir -p "$OUT/$name"
  python -m common.grpo.train_grpo --backend neuron --platform trainium --task cholect50 \
    --model "$MODEL" --data "$CODE/data/cholect50/parquet" \
    --rollout-batch-size 16 --n-samples 8 --temperature 0.7 \
    --max-prompt-len 2048 --max-new-tokens 256 --lr 1e-5 --kl-beta 0.04 \
    --lora-rank 16 --dtype bf16 --grad-ckpt --gen-micro-batch 8 --train-micro-batch 1 \
    --out-dir "$OUT/$name" "$@" 2>&1 | tee "$OUT/$name/console.log"
  return "${PIPESTATUS[0]}"
}

if want rl1; then
  # Mini GRPO step: the full code path (sample -> reward -> group advantages -> clipped loss + KL
  # -> LoRA update) on 2 frames x 4 samples, <= 64 tokens: one sampler batch, ~5-15 min.
  stage rl1
  t0=$(date +%s)
  run_grpo rl1 --num-steps 1 --eval-split "" --rollout-batch-size 2 --n-samples 4 \
    --max-new-tokens 64 --gen-micro-batch 8 --train-micro-batch 1 \
    || { log "rl1 FAILED after $(( $(date +%s) - t0 )) s"; exit 4; }
  log "rl1 done in $(( $(date +%s) - t0 )) s"
fi

if want rl1full; then
  # One full-size GRPO step (16 frames x 8 samples, 256 tokens): ~1-2.5 h with the static sampler.
  stage rl1full
  t0=$(date +%s)
  run_grpo rl1full --num-steps 1 --eval-split "" || { log "rl1full FAILED after $(( $(date +%s) - t0 )) s"; exit 6; }
  log "rl1full done in $(( $(date +%s) - t0 )) s"
fi

if want full; then
  stage full
  t0=$(date +%s)
  run_grpo full --eval-split val || { log "full FAILED after $(( $(date +%s) - t0 )) s"; exit 5; }
  log "full done in $(( $(date +%s) - t0 )) s"
fi

stage complete
log "job $RUN_ID complete"

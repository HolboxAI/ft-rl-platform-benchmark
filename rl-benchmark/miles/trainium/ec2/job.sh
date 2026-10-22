#!/usr/bin/env bash
# Miles on Trainium (torch-neuronx): the job the per-boot runner executes on the trn instance.
# Environment comes from control/job.env (written by submit.sh).
#
# Miles has no Neuron platform and its rollout engine (SGLang) is CUDA-only, so a
# Miles RL run cannot start here. This job measures exactly how far Miles gets:
#   env      record the Neuron venv (torch, torch-xla, torch-neuronx) and neuron-ls
#   install  Miles @ MILES_REF with --no-deps + its light pure-Python deps, never touching torch
#   sglang   what installing SGLang would pull (dry-run resolution, nothing installed)
#   probe    miles/scripts/probe_miles.py --backend neuron (6 stages, JSON result)
set -uo pipefail
: "${S3_ROOT:?}" "${RUN_ID:?}" "${MILES_REF:?}"
DATA=/data
CODE=$DATA/code/rl-benchmark
OUT=$DATA/runs/$RUN_ID
MILES=$DATA/miles
mkdir -p "$OUT" /data/state
export HOME=${HOME:-/root} PYTHONUNBUFFERED=1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
stage() { echo "$1" > /data/state/current_stage; log "=== stage $1"; }
sync_up() { aws s3 sync --only-show-errors "$OUT/" "$S3_ROOT/runs/$RUN_ID/out/" || true; }
trap sync_up EXIT

aws s3 sync --delete --only-show-errors "$S3_ROOT/code/" "$CODE/"

stage env
# Venv naming differs by DLAMI: aws_neuronx_venv_pytorch_<ver> (older) or
# aws_neuronx_venv_torch_neuronx_<ver> (torch-neuronx 2.14 image).
ls -d /opt/aws_neuronx_venv_* 2>/dev/null | tee "$OUT/venvs.txt"
VENV=$(ls -d /opt/aws_neuronx_venv_torch_neuronx_* /opt/aws_neuronx_venv_pytorch* 2>/dev/null | sort -V | tail -1)
log "Neuron venv: ${VENV:-none found}"
[ -n "$VENV" ] || { log "FATAL: no Neuron PyTorch venv under /opt"; exit 2; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PATH="/opt/aws/neuron/bin:$PATH"   # neuron-ls, neuron-top
command -v python >/dev/null || { log "FATAL: python not on PATH after activating $VENV"; exit 2; }
{
  echo "# $(date -u +%FT%TZ) $(hostname) venv=$VENV"
  python --version
  neuron-ls 2>&1 || true
  pip list 2>/dev/null | grep -iE "^(torch|torch-xla|torch-neuronx|neuronx|libneuronxla|transformers|ray|jax)" || true
} | tee "$OUT/environment.txt"
# Freeze the Neuron-built torch stack so no later pip step can replace it.
pip freeze 2>/dev/null | grep -iE "^(torch|torch-xla|torch-neuronx|torchvision|neuronx-cc|libneuronxla)==" > "$OUT/constraints.txt"
cat "$OUT/constraints.txt"
sync_up

stage install
[ -d "$MILES/.git" ] || git clone https://github.com/radixark/miles "$MILES"
git -C "$MILES" checkout "$MILES_REF" 2>&1 | tail -1
git -C "$MILES" log -1 --format='miles %h %cs' | tee -a "$OUT/environment.txt"
pip install --no-deps -e "$MILES" 2>&1 | tail -2
pip install -c "$OUT/constraints.txt" omegaconf pybase64 orjson httpx "ray[default]>=2.56,<2.59" qwen_vl_utils \
  "transformers==5.19.0" 2>&1 | tail -3 | tee "$OUT/install_light_deps.txt"
sync_up

stage sglang
# Resolution only: shows which CUDA wheels SGLang would bring in. --dry-run installs nothing.
pip install --dry-run --ignore-installed sglang 2>&1 | tail -15 | tee "$OUT/sglang_dry_run.txt"
sync_up

stage probe
python "$CODE/miles/scripts/probe_miles.py" --backend neuron --miles-src "$MILES" \
  --out "$OUT/probe_trainium.json" 2>&1 | tee "$OUT/probe_console.log"
sync_up

stage complete
log "job $RUN_ID complete"

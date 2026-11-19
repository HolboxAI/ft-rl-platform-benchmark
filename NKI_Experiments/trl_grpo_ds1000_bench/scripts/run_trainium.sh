#!/usr/bin/env bash
# Trainium runs (trn1 / trn2 / trn3) with torch-neuronx. Uses ONE NeuronCore so numbers are
# per-core and comparable with one L40S; set NEURON_RT_VISIBLE_CORES to use more.
#
#   PLATFORM=trn2.3xlarge INSTANCE_PRICE_USD_HR=... REPEATS=3 bash scripts/run_trainium.sh
#
# Env:
#   PLATFORM               trn1.2xlarge | trn2.3xlarge | trn2.48xlarge | trn3.* (label only)
#   REPEATS                runs per variant (default 1)
#   VARIANTS               space-separated nki:sync pairs
#                          (default "none:phase rmsnorm:phase none:step rmsnorm:step")
#   NEURON_VENV            venv with torch-neuronx (default: first /opt/aws_neuronx_venv_pytorch_*)
#   INSTANCE_PRICE_USD_HR  on-demand $/hr (for cost columns)
#   S3_RESULTS             optional s3://... prefix to sync results to
NEURON_VENV="${NEURON_VENV:-$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | grep -v inference | head -1)}"
if [ -n "$NEURON_VENV" ] && [ -f "$NEURON_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$NEURON_VENV/bin/activate"
fi
source "$(dirname "$0")/_common.sh"

PLATFORM="${PLATFORM:?set PLATFORM, e.g. trn2.3xlarge}"
REPEATS="${REPEATS:-1}"
VARIANTS="${VARIANTS:-none:phase rmsnorm:phase none:step rmsnorm:step}"
export INSTANCE_TYPE="${INSTANCE_TYPE:-$PLATFORM}"

# --- Neuron runtime / compiler settings (recorded in run_info.csv)
export NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---model-type=transformer --retry_failed_compilation}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-$HOME/neuron_compile_cache}"
export BENCH_BACKEND=neuron
# trn2/trn3 default to logical NeuronCore config 2 (two physical cores act as one).
case "$PLATFORM" in trn2*|trn3*) export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}";; esac

ensure_subset
python -m kernels.kernel_microbench --platform "$PLATFORM" --output-root "$RESULTS_DIR"

for r in $(seq 1 "$REPEATS"); do
  for v in $VARIANTS; do
    run_variant neuron "$PLATFORM" "${v%%:*}" "${v##*:}"
  done
done

python -m metrics.compare_runs --results "$RESULTS_DIR" || true
maybe_upload

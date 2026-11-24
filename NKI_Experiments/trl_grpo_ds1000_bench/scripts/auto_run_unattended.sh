#!/usr/bin/env bash
# Unattended Trainium benchmark: needs no laptop / SSH session once started.
#
#   wait for setup (and optionally a running smoke test) -> 4 benchmark variants
#   -> kernel microbench -> comparison tables. After every stage a tarball is PUT to S3 via
#   presigned URLs (the instances have no IAM role), then the instance shuts itself down.
#   A hard cap (MAX_HOURS) shuts it down whatever happens.
#
# Start it detached on the instance:
#   PLATFORM=trn2.48xlarge INSTANCE_PRICE_USD_HR=8.60 URLS=~/upload_urls.txt \
#     setsid nohup bash scripts/auto_run_unattended.sh </dev/null >/dev/null 2>&1 &
#
# URLS file: one "<label> <presigned PUT url>" per line, labels:
#   0_smoke_2steps 1_nki-none_sync-phase 2_nki-rmsnorm_sync-phase 3_nki-none_sync-step
#   4_nki-rmsnorm_sync-step 5_kernels_and_comparison 6_final_all_results
# Missing labels are skipped (no upload). Progress: ~/auto_run.log
#
# Env: PLATFORM (required), INSTANCE_PRICE_USD_HR, URLS (default ~/upload_urls.txt),
#      WAIT_FOR_SMOKE=1 (wait for ~/smoke.log SMOKE_EXIT= and stop if it failed),
#      MAX_HOURS (default 7), SHUTDOWN=1 (0 = leave the instance running at the end),
#      VARIANTS (default "none:phase rmsnorm:phase none:step rmsnorm:step")
set -u
PLATFORM="${PLATFORM:?set PLATFORM}"
URLS="${URLS:-$HOME/upload_urls.txt}"
MAX_HOURS="${MAX_HOURS:-7}"
SHUTDOWN="${SHUTDOWN:-1}"
VARIANTS="${VARIANTS:-none:phase rmsnorm:phase none:step rmsnorm:step}"
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES="$BENCH/results"
LOG="$HOME/auto_run.log"
exec >>"$LOG" 2>&1
echo "=== auto_run start $(date -u) platform=$PLATFORM"

stop_instance() { [ "$SHUTDOWN" = "1" ] && { sync; sudo shutdown -h now; }; }
( sleep $((MAX_HOURS * 3600)); echo "=== ${MAX_HOURS}h cap reached $(date -u)"; stop_instance ) &

upload() {   # upload <label> <path...>
  local label=$1; shift
  local url; url=$(awk -v l="$label" '$1==l{print $2}' "$URLS" 2>/dev/null)
  [ -z "$url" ] && { echo "=== upload $label: no URL, skipped"; return; }
  local tgz=/tmp/$label.tar.gz
  tar -czf "$tgz" --exclude='*/trainer_out/*' "$@" "$LOG" 2>/dev/null
  local code; code=$(curl -s -o /dev/null -w "%{http_code}" -X PUT --upload-file "$tgz" "$url")
  echo "=== upload $label -> HTTP $code ($(du -h "$tgz" | cut -f1)) $(date -u)"
}

finish() {
  upload 6_final_all_results "$RES" $( [ -d ~/smoke ] && echo ~/smoke ) $( [ -f ~/smoke.log ] && echo ~/smoke.log ) \
         $( [ -f ~/setup.log ] && echo ~/setup.log )
  echo "=== auto_run done $(date -u)"
  stop_instance; exit 0
}

# 0. wait for instance setup (bootstrap user-data writes ~/.setup_ok)
if [ -f ~/bootstrap.log ]; then
  while [ ! -f ~/.setup_ok ]; do
    grep -q "SETUP FAILED\|FAIL download\|FAIL checksum" ~/bootstrap.log && { echo "=== setup failed"; finish; }
    sleep 30
  done
fi

# 1. optional: wait for a smoke test that is already running
if [ "${WAIT_FOR_SMOKE:-0}" = "1" ]; then
  while ! grep -q SMOKE_EXIT= ~/smoke.log 2>/dev/null; do sleep 30; done
  rc=$(grep -o 'SMOKE_EXIT=[0-9]*' ~/smoke.log | cut -d= -f2)
  echo "=== smoke exit=$rc $(date -u)"
  upload 0_smoke_2steps ~/smoke ~/smoke.log
  [ "$rc" != "0" ] && { echo "=== smoke failed, not running full variants"; finish; }
fi

# 2. environment (same settings as scripts/run_trainium.sh)
NEURON_VENV="${NEURON_VENV:-$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | grep -v inference | head -1)}"
# shellcheck disable=SC1091
source "$NEURON_VENV/bin/activate"
cd "$BENCH"
export INSTANCE_TYPE="$PLATFORM" INSTANCE_PRICE_USD_HR="${INSTANCE_PRICE_USD_HR:-}"
export NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0}" BENCH_BACKEND=neuron
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---model-type=transformer --retry_failed_compilation}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/var/tmp/neuron-compile-cache}"
case "$PLATFORM" in trn2*|trn3*) export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}";; esac
export REWARD_PYTHON="${REWARD_PYTHON:-$HOME/ds1000-env/bin/python}"
# shellcheck disable=SC1091
source scripts/_common.sh
set +e

# 3. benchmark variants, one upload each
ensure_subset
i=1
for v in $VARIANTS; do
  nki=${v%%:*}; sync=${v##*:}
  echo "=== variant $v start $(date -u)"
  run_variant neuron "$PLATFORM" "$nki" "$sync"
  d=$(ls -dt "$RES"/"$PLATFORM"_neuron_nki-"$nki"_sync-"$sync"_* 2>/dev/null | head -1)
  upload "${i}_nki-${nki}_sync-${sync}" "$d"
  i=$((i + 1))
done

# 4. kernel microbench + comparison tables
echo "=== kernels start $(date -u)"
timeout 5400 python -m kernels.kernel_microbench --platform "$PLATFORM" --output-root "$RES"
python -m metrics.compare_runs --results "$RES"
upload 5_kernels_and_comparison "$RES"/kernels_"$PLATFORM"_* "$RES"/comparison
finish

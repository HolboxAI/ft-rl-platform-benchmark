#!/usr/bin/env bash
# GPU baseline on g6e (NVIDIA L40S, 48 GB). Uses one GPU.
#
#   INSTANCE_PRICE_USD_HR=1.861 REPEATS=3 bash scripts/run_g6e.sh
#
# Env:
#   PLATFORM               label used in results (default g6e.xlarge)
#   REPEATS                runs per variant (default 1; use 3 and report medians)
#   INSTANCE_PRICE_USD_HR  on-demand $/hr on the day of the test (for cost columns)
#   S3_RESULTS             optional s3://... prefix to sync results to
source "$(dirname "$0")/_common.sh"

PLATFORM="${PLATFORM:-g6e.xlarge}"
REPEATS="${REPEATS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INSTANCE_TYPE="${INSTANCE_TYPE:-$PLATFORM}"

ensure_subset
python -m kernels.kernel_microbench --platform "$PLATFORM" --output-root "$RESULTS_DIR"

for r in $(seq 1 "$REPEATS"); do
  run_variant cuda "$PLATFORM" none phase   # per-phase attribution
  run_variant cuda "$PLATFORM" none step    # end-to-end step time (no per-phase syncs)
done

python -m metrics.compare_runs --results "$RESULTS_DIR" || true
maybe_upload

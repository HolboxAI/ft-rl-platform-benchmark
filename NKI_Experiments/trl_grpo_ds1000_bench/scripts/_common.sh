#!/usr/bin/env bash
# Shared helpers for run_g6e.sh / run_trainium.sh. Source, don't execute.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/results}"
cd "$BENCH_DIR"
export PYTHONPATH="$BENCH_DIR:${PYTHONPATH:-}"
# DS-1000 tests run in this interpreter (separate env with the DS-1000 library pins)
export REWARD_PYTHON="${REWARD_PYTHON:-$HOME/ds1000-env/bin/python}"
[ -x "$REWARD_PYTHON" ] || { echo "REWARD_PYTHON not found: $REWARD_PYTHON (run scripts/setup_reward_env.sh)"; exit 1; }

ensure_subset() {
  if [ ! -f data/ds1000_subset32.jsonl ]; then
    python -m data.prepare_ds1000_subset --source "${DS1000_JSONL:-$BENCH_DIR/../data/ds1000/test.jsonl}"
  fi
}

# run_variant <backend cuda|neuron> <platform> <nki_ops> <sync_mode>
run_variant() {
  local backend=$1 platform=$2 nki=$3 sync=$4
  local stamp; stamp=$(date -u +%Y%m%dT%H%M%SZ)
  local name="${platform}_${backend}_nki-${nki}_sync-${sync}_${stamp}"
  local dir="$RESULTS_DIR/$name"
  mkdir -p "$dir"
  python -m metrics.device_monitor --backend "$backend" --out "$dir/device_util.csv" &
  local mon=$!
  set +e
  python -m train.grpo_ds1000_bench --platform "$platform" --nki-ops "$nki" --sync-mode "$sync" \
      --run-name "$name" --output-root "$RESULTS_DIR" 2>&1 | tee "$dir/train.log"
  local rc=${PIPESTATUS[0]}
  set -e
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  # re-summarise now that device_util.csv is complete
  python -m metrics.summarize_run "$dir" || true
  echo "[run] $name exit=$rc"
}

maybe_upload() {
  if [ -n "${S3_RESULTS:-}" ]; then
    aws s3 sync "$RESULTS_DIR" "$S3_RESULTS" --exclude "*/trainer_out/*"
  fi
}

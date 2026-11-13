#!/usr/bin/env bash
# Put the CholecT50 inputs on a VM (run on BOTH VMs; the rollout VM needs only the model).
#
#   DATA_GCS=gs://<bucket>/cholec bash ~/tpu_rl_benchmark/skyrl-tpu/setup/fetch_data.sh   # from GCS (preferred)
#   bash ~/tpu_rl_benchmark/skyrl-tpu/setup/fetch_data.sh                                 # from S3; needs AWS read creds
#
# Same inputs as the GPU run and every other leg:
#   SFT checkpoint  cholec-sft-qwen3.5-4b (Tinker SFT, merged; 9.9 GB)  -> ~/models/cholec-sft-qwen3.5-4b
#   verl parquet    256 train / 32 val frames                            -> ~/cholec/verl_parquet
#   SkyRL parquet   converted with skyrl_cholect50/data.py (GPU run's)   -> ~/cholec/skyrl_parquet
# The parquet hashes are checked against the copy the GPU SkyRL control run used.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_CKPT=${SFT_CKPT:-$HOME/models/cholec-sft-qwen3.5-4b}
VERL_DIR=${VERL_DIR:-$HOME/cholec/verl_parquet}
SKYRL_DIR=${SKYRL_PARQUET_DIR:-$HOME/cholec/skyrl_parquet}
PY=${PY:-$HOME/.venv-skyrl-tpu/bin/python}
MODEL_ONLY=${MODEL_ONLY:-0}
mkdir -p "$SFT_CKPT" "$VERL_DIR" "$SKYRL_DIR"

if [ -n "${DATA_GCS:-}" ]; then
  gcloud storage rsync --recursive "$DATA_GCS/models/cholec-sft-qwen3.5-4b" "$SFT_CKPT"
  [ "$MODEL_ONLY" = 1 ] || gcloud storage rsync --recursive "$DATA_GCS/parquet" "$VERL_DIR"
else
  command -v aws >/dev/null || { echo "no DATA_GCS and no aws CLI; see README 'Data'"; exit 1; }
  aws s3 sync --only-show-errors s3://stanford-train-data/models/cholec-sft-qwen3.5-4b/ "$SFT_CKPT/"
  [ "$MODEL_ONLY" = 1 ] || aws s3 sync --only-show-errors s3://stanford-train-data/cholec/parquet/ "$VERL_DIR/"
fi
ls -la "$SFT_CKPT" | head -20
[ "$MODEL_ONLY" = 1 ] && exit 0

(cd "$VERL_DIR" && sha256sum -c - <<'EOF'
7463bb11bd6518b469f0cfe248218090fe6896b6e10f70721ee7a0910fb1bd10  train.parquet
5c6f3191fedd088b9dbd2c1c8e3e2089e075259cdc2ed01a19010686ef5c4c8c  val.parquet
EOF
)
"$PY" "$ROOT/skyrl_cholect50/data.py" --verl_dir "$VERL_DIR" --out_dir "$SKYRL_DIR"

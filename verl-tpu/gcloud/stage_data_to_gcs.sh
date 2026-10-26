#!/usr/bin/env bash
# One-time: copy the CholecT50 inputs from AWS S3 into a GCS bucket next to the TPUs. Runs on the laptop.
#
#   aws login   (or any AWS credentials that can read s3://stanford-train-data)
#   bash gcloud/stage_data_to_gcs.sh                 # bucket gs://<project>-cholec in the TPU zone's region
#   BUCKET=gs://my-bucket bash gcloud/stage_data_to_gcs.sh
#
# What is copied (the only CholecT50 data any leg uses; the raw CholecT50 videos are not needed):
#   s3://stanford-train-data/cholec/parquet/                    -> $BUCKET/cholec/parquet/            (256 train / 32 val frames, images inline)
#   s3://stanford-train-data/models/cholec-sft-qwen3.5-4b/      -> $BUCKET/cholec/models/cholec-sft-qwen3.5-4b/  (9.9 GB)
# Then on each VM:  DATA_GCS=$BUCKET/cholec bash setup/fetch_data.sh
#
# The parquet hashes are checked before upload against the copy the GPU control run used.
# Data goes laptop -> GCS (about 10 GB through this machine), so run it on a good connection.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config/tpu.env.sh"
REGION=${REGION:-${ZONE%-*}}
BUCKET=${BUCKET:-gs://$PROJECT-cholec}
STAGE=${STAGE:-$HOME/cholec_stage}
G=("$GCLOUD" --project "$PROJECT")

aws sts get-caller-identity > /dev/null || { echo "no AWS credentials: run 'aws login' first"; exit 1; }
mkdir -p "$STAGE/parquet" "$STAGE/models/cholec-sft-qwen3.5-4b"
aws s3 sync --only-show-errors s3://stanford-train-data/cholec/parquet/ "$STAGE/parquet/"
(cd "$STAGE/parquet" && shasum -a 256 -c - <<'EOF'
7463bb11bd6518b469f0cfe248218090fe6896b6e10f70721ee7a0910fb1bd10  train.parquet
5c6f3191fedd088b9dbd2c1c8e3e2089e075259cdc2ed01a19010686ef5c4c8c  val.parquet
EOF
)
aws s3 sync --only-show-errors s3://stanford-train-data/models/cholec-sft-qwen3.5-4b/ "$STAGE/models/cholec-sft-qwen3.5-4b/"

if ! "${G[@]}" storage buckets describe "$BUCKET" > /dev/null 2>&1; then
  "${G[@]}" storage buckets create "$BUCKET" --location "$REGION" --uniform-bucket-level-access
fi
"${G[@]}" storage rsync --recursive "$STAGE/parquet" "$BUCKET/cholec/parquet"
"${G[@]}" storage rsync --recursive "$STAGE/models/cholec-sft-qwen3.5-4b" "$BUCKET/cholec/models/cholec-sft-qwen3.5-4b"
"${G[@]}" storage ls -l "$BUCKET/cholec/**" | tail -5
echo "staged. On each VM: DATA_GCS=$BUCKET/cholec bash ~/tpu_rl_benchmark/setup/fetch_data.sh"

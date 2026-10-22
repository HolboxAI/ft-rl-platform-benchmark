#!/bin/bash
# Fetch the CholecT50 inputs every leg uses (run from the repo root on the accelerator host):
#   - SFT checkpoint  s3://stanford-train-data/models/cholec-sft-qwen3.5-4b -> $SFT_CKPT (9.9 GB)
#   - verl parquet    s3://stanford-train-data/cholec/parquet             -> data/cholect50/parquet (256 / 32 frames)
# Needs read access to the bucket (EC2: the ec2-sft-trainer instance profile; elsewhere: AWS credentials).
set -euo pipefail
cd "$(dirname "$0")/.."
SFT_CKPT=${SFT_CKPT:-$HOME/models/cholec-sft-qwen3.5-4b}
mkdir -p "$SFT_CKPT" data/cholect50/parquet
aws s3 sync --only-show-errors s3://stanford-train-data/models/cholec-sft-qwen3.5-4b/ "$SFT_CKPT/"
aws s3 sync --only-show-errors s3://stanford-train-data/cholec/parquet/ data/cholect50/parquet/
# Hashes of the copy used by the SkyRL GPU control run (2026-10-22); a mismatch means different frames.
cat > /tmp/cholect50.sha256 <<'EOF'
7463bb11bd6518b469f0cfe248218090fe6896b6e10f70721ee7a0910fb1bd10  data/cholect50/parquet/train.parquet
5c6f3191fedd088b9dbd2c1c8e3e2089e075259cdc2ed01a19010686ef5c4c8c  data/cholect50/parquet/val.parquet
EOF
sha256sum -c /tmp/cholect50.sha256
ls -la "$SFT_CKPT" | head -20

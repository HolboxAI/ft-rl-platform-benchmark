#!/bin/bash
# Download the CholecT50 train/val JSONL (Miles format, base64 images) from S3.
set -euo pipefail
mkdir -p /home/ubuntu/bench
cd /home/ubuntu/bench

# shellcheck disable=SC1091
source /home/ubuntu/urls.sh

curl -sS -f -o cholect50_train.jsonl "$URL_cholect50_train_jsonl"
curl -sS -f -o cholect50_val.jsonl "$URL_cholect50_val_jsonl"

ls -la /home/ubuntu/bench

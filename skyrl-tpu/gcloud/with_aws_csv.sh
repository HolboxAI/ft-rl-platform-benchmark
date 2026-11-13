#!/usr/bin/env bash
# Run one command with AWS keys taken from an IAM access-key CSV (columns
# "Access key ID", "Secret access key"). The keys live only in that command's
# environment: nothing is written to ~/.aws, to this repo, or to any log.
#
#   AWS_KEYS_CSV=~/Documents/RL\ benchmark/stanford_gpu_accessKeys.csv \
#     bash gcloud/with_aws_csv.sh aws s3 ls s3://stanford-train-data/cholec/parquet/
#   AWS_KEYS_CSV=... bash gcloud/with_aws_csv.sh bash gcloud/stage_data_to_gcs.sh
set -euo pipefail
CSV=${AWS_KEYS_CSV:?set AWS_KEYS_CSV to the access-key CSV}
[ $# -gt 0 ] || { echo "usage: AWS_KEYS_CSV=... $0 <command> [args...]"; exit 1; }
eval "$(python3 - "$CSV" <<'EOF'
import csv, shlex, sys
row = next(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8-sig")))
print("export AWS_ACCESS_KEY_ID=" + shlex.quote(row["Access key ID"].strip()))
print("export AWS_SECRET_ACCESS_KEY=" + shlex.quote(row["Secret access key"].strip()))
EOF
)"
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
exec "$@"

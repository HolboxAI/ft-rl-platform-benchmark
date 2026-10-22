#!/usr/bin/env bash
# Upload this repo + the GRPO-loop CholecT50 job to S3, then start (or reboot) the instance.
#   trainium/ec2/submit.sh [--run-id ID] [--stages "env install data smoke rl1"] [--no-start] [--force]
# Re-using a RUN_ID resumes on the same disk (finished phases are skipped).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
# shellcheck source=ec2.env
source "$HERE/ec2.env"
AWS=(aws --profile "$AWS_PROFILE" --region "$AWS_REGION")

RUN_ID="grpo-trn-$(date -u +%Y%m%d-%H%M)"
START=1; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID=$2; shift 2 ;;
    --stages) STAGES=$2; shift 2 ;;
    --no-start) START=0; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

if [ "$START" = 1 ] && [ "$FORCE" = 0 ]; then
  current=$("${AWS[@]}" s3 cp "$S3_ROOT/control/status.json" - 2>/dev/null || true)
  if echo "$current" | grep -q '"state": "RUNNING"'; then
    echo "a job is RUNNING: $current"; echo "use --force to reboot into the new job"; exit 1
  fi
fi

# Local gates: the probe compiles, the job parses.
python3 -m py_compile "$REPO/common/smoke_test.py" "$REPO/common/neuron_compat.py" "$REPO"/common/grpo/*.py
bash -n "$HERE/job.sh"

"${AWS[@]}" s3 sync --delete --only-show-errors "$REPO/" "$S3_ROOT/code/" \
  --exclude ".DS_Store" --exclude "*/__pycache__/*" --exclude "*/.instance_id" --exclude ".venv*" \
  --exclude "*/logs/*" --exclude "data/cholect50/parquet/*"

JOB_ENV=$(mktemp)
for var in RUN_ID S3_ROOT STAGES MODEL_S3_URI PARQUET_S3_URI TRANSFORMERS_VERSION INSTANCE_TYPE PRICE_PER_HOUR_USD; do
  printf '%s=%q\n' "$var" "${!var}"
done > "$JOB_ENV"
"${AWS[@]}" s3 cp --only-show-errors "$JOB_ENV" "$S3_ROOT/control/job.env"
"${AWS[@]}" s3 cp --only-show-errors "$HERE/job.sh" "$S3_ROOT/control/job.sh"
rm -f "$JOB_ENV"
echo "submitted RUN_ID=$RUN_ID stages=$STAGES"

[ "$START" = 1 ] || { echo "not starting the instance (--no-start)"; exit 0; }
[ -f "$HERE/.instance_id" ] || { echo "no instance yet: run trainium/ec2/launch_instance.sh"; exit 0; }
ID=$(cat "$HERE/.instance_id")
state=$("${AWS[@]}" ec2 describe-instances --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
case "$state" in
  stopped) "${AWS[@]}" ec2 start-instances --instance-ids "$ID" >/dev/null; echo "starting $ID" ;;
  running) "${AWS[@]}" ec2 reboot-instances --instance-ids "$ID"; echo "rebooting $ID into the new job" ;;
  *) echo "instance $ID is '$state'; it will pick the job up on its next boot" ;;
esac

#!/usr/bin/env bash
# Check the GPU control run from the laptop.
#   skyrl/gpu_control/ec2/status.sh [--lines N] [--fetch]
# --fetch copies results into skyrl/metrics/ and logs into skyrl/logs/<RUN_ID>/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
# shellcheck source=ec2.env
source "$HERE/ec2.env"
AWS=(aws --profile "$AWS_PROFILE" --region "$AWS_REGION")
LINES=25; FETCH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --lines) LINES=$2; shift 2 ;;
    --fetch) FETCH=1; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

status=$("${AWS[@]}" s3 cp "$S3_ROOT/control/status.json" - 2>/dev/null || echo '{}')
echo "status:   $status"
if [ -f "$HERE/.instance_id" ]; then
  echo "instance: $(cat "$HERE/.instance_id") $("${AWS[@]}" ec2 describe-instances --instance-ids "$(cat "$HERE/.instance_id")" \
    --query 'Reservations[0].Instances[0].[State.Name,InstanceType,LaunchTime]' --output text)"
fi
run_id=$(echo "$status" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("run_id",""))')
[ -n "$run_id" ] || exit 0
logf=$(mktemp)
if "${AWS[@]}" s3 cp --only-show-errors "$S3_ROOT/runs/$run_id/job.log" "$logf" 2>/dev/null; then
  echo "--- stages ---"; grep -E '=== stage|server up|preset .* (done|FAILED)|probe FAILED|job .* complete' "$logf" | tail -n 12 || true
  echo "--- probe / eval / step lines ---"; grep -aE '\[probe\]|\[eval\]|\[config\]' "$logf" | tail -n 12 || true
  echo "--- log tail ---"; tail -n "$LINES" "$logf"
fi
rm -f "$logf"
if [ "$FETCH" = 1 ]; then
  mkdir -p "$REPO/skyrl/logs/$run_id"
  "${AWS[@]}" s3 sync --only-show-errors "$S3_ROOT/runs/$run_id/out/" "$REPO/skyrl/logs/$run_id/"
  "${AWS[@]}" s3 cp --only-show-errors "$S3_ROOT/runs/$run_id/job.log" "$REPO/skyrl/logs/$run_id/job.log" || true
  for p in tinker skyrl-grpo; do
    f="$REPO/skyrl/logs/$run_id/$p/final_results.json"
    [ -f "$f" ] && cp "$f" "$REPO/skyrl/metrics/cholect50_rl_gpu_${p}_final_results.json" && echo "results: skyrl/metrics/cholect50_rl_gpu_${p}_final_results.json"
  done
  echo "fetched into skyrl/logs/$run_id"
fi

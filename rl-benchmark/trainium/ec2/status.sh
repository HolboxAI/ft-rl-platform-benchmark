#!/usr/bin/env bash
# Check the GPU control run from the laptop.
#   trainium/ec2/status.sh [--lines N] [--fetch]
# --fetch copies results + logs into trainium/logs/<RUN_ID>/ and trainium/metrics/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
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
  echo "--- stages ---"; grep -aE '=== stage|\[grpo\]|\[step |\[eval\]|\[neuron_compat\]|FATAL|job .* complete' "$logf" | tail -n 12 || true
  echo "--- probe / eval / step lines ---"; grep -aE "'loss'|'reward'|\\[check\\]|\\[eval\\]" "$logf" | tail -n 12 || true
  echo "--- log tail ---"; tail -n "$LINES" "$logf"
fi
rm -f "$logf"
if [ "$FETCH" = 1 ]; then
  mkdir -p "$REPO/trainium/logs/$run_id"
  "${AWS[@]}" s3 sync --only-show-errors "$S3_ROOT/runs/$run_id/out/" "$REPO/trainium/logs/$run_id/"
  "${AWS[@]}" s3 cp --only-show-errors "$S3_ROOT/runs/$run_id/job.log" "$REPO/trainium/logs/$run_id/job.log" || true
  for f in smoke.json rl1/final_results.json full/final_results.json; do
    [ -f "$REPO/trainium/logs/$run_id/$f" ] && echo "result: trainium/logs/$run_id/$f"
  done
  [ -f "$REPO/trainium/logs/$run_id/full/final_results.json" ] && cp "$REPO/trainium/logs/$run_id/full/final_results.json" \
    "$REPO/trainium/metrics/cholect50_rl_final_results.json"
  echo "fetched into trainium/logs/$run_id"
fi

#!/usr/bin/env bash
# One-time: create the GPU instance. On its first boot it runs whatever job
# submit.sh uploaded, so run `submit.sh --no-start` first.
#   ec2/launch_instance.sh [--dry-run]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=ec2.env
source "$HERE/ec2.env"
AWS=(aws --profile "$AWS_PROFILE" --region "$AWS_REGION")
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

if [ -f "$HERE/.instance_id" ]; then
  echo "instance already recorded: $(cat "$HERE/.instance_id") (delete hf-trl/trainium/ec2/.instance_id to launch another)"; exit 1
fi
"${AWS[@]}" s3 ls "$S3_ROOT/control/job.sh" >/dev/null \
  || { echo "no job at $S3_ROOT/control/job.sh: run hf-trl/trainium/ec2/submit.sh --no-start first"; exit 1; }

AMI=${AMI_ID:-$("${AWS[@]}" ssm get-parameter --name "$DLAMI_SSM_PARAM" --query Parameter.Value --output text)}
echo "AMI: $AMI  type: $INSTANCE_TYPE  market: $MARKET"

UD=$(mktemp)
"$HERE/render_user_data.sh" "$UD"   # MIME multipart: user_data.sh runs on every boot

extra=()
if [ "$MARKET" = spot ]; then
  extra+=(--instance-market-options
          'MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}')
fi
[ -n "$SECURITY_GROUP_ID" ] && extra+=(--security-group-ids "$SECURITY_GROUP_ID")
[ -n "$KEY_NAME" ] && extra+=(--key-name "$KEY_NAME")

# GPU capacity is per availability zone. With no SUBNET_ID set, try the default-VPC subnet
# of every zone that offers INSTANCE_TYPE until one has capacity.
if [ -n "$SUBNET_ID" ]; then
  subnets=("$SUBNET_ID")
else
  zones=$("${AWS[@]}" ec2 describe-instance-type-offerings --location-type availability-zone \
    --filters "Name=instance-type,Values=$INSTANCE_TYPE" --query 'InstanceTypeOfferings[].Location' --output text)
  subnets=()
  for z in $zones; do
    s=$("${AWS[@]}" ec2 describe-subnets --filters "Name=default-for-az,Values=true" "Name=availability-zone,Values=$z" \
      --query 'Subnets[0].SubnetId' --output text)
    [ "$s" != "None" ] && subnets+=("$s")
  done
fi

ID=""
for subnet in "${subnets[@]}"; do
  echo "trying subnet $subnet ..."
  if out=$("${AWS[@]}" ec2 run-instances $DRY \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --count 1 --subnet-id "$subnet" \
    --iam-instance-profile "Name=$IAM_INSTANCE_PROFILE" \
    --instance-initiated-shutdown-behavior stop \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true}" \
    --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
    --user-data "file://$UD" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=project,Value=accel-bench-trl}]" \
    ${extra[@]+"${extra[@]}"} \
    --query 'Instances[0].InstanceId' --output text 2>&1); then
    ID=$out
    break
  fi
  echo "  $out" | tail -1
  case "$out" in *InsufficientInstanceCapacity*|*Unsupported*) continue ;; *) rm -f "$UD"; exit 1 ;; esac
done
rm -f "$UD"
[ -n "$ID" ] || { echo "no capacity for $INSTANCE_TYPE in any zone; retry later or change INSTANCE_TYPE"; exit 1; }
echo "$ID" > "$HERE/.instance_id"
echo "launched $ID; watch it with hf-trl/trainium/ec2/status.sh"

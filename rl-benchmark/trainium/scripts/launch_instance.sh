#!/bin/bash
# Launch a Trainium instance on the AWS Neuron Deep Learning AMI.
#
#   INSTANCE_TYPE=trn1.2xlarge KEY_NAME=my-key SUBNET_ID=subnet-... SG_ID=sg-... \
#     bash trainium/scripts/launch_instance.sh
#
# CholecT50 (Qwen3.5-4B VLM, LoRA 16): trn1.2xlarge — 1 Trainium1 chip, 2 NeuronCores, 32 GB HBM (16 GB/core).
# If one NeuronCore runs out of memory: trn2 (96 GB HBM / chip) — see config/TRAINIUM_CONFIG.md.
#
# Trainium is not offered in every region/AZ. Check quota "Running On-Demand Trn
# instances" (vCPU-based) first; new accounts often start at 0.
set -euo pipefail

INSTANCE_TYPE=${INSTANCE_TYPE:-trn1.2xlarge}
REGION=${AWS_REGION:-us-east-1}
VOLUME_GB=${VOLUME_GB:-300}
NAME=${NAME:-trainium-grpo-bench}
: "${KEY_NAME:?set KEY_NAME}"
: "${SUBNET_ID:?set SUBNET_ID (an AZ that offers ${INSTANCE_TYPE})}"
: "${SG_ID:?set SG_ID}"

# AWS publishes the latest Neuron DLAMI id as a public SSM parameter. Override
# with AMI_ID to pin a specific Neuron SDK release (recommended for reruns).
SSM_PARAM=${SSM_PARAM:-/aws/service/neuron/dlami/multi-framework/ubuntu-22.04/latest/image_id}
AMI_ID=${AMI_ID:-$(aws ssm get-parameter --region "$REGION" --name "$SSM_PARAM" --query Parameter.Value --output text)}
echo "AMI: $AMI_ID ($SSM_PARAM)"

aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}}]" \
  --query 'Instances[0].InstanceId' --output text | tee "$(dirname "$0")/.instance_id"

echo "Remember to stop/terminate: aws ec2 terminate-instances --region $REGION --instance-ids \$(cat $(dirname "$0")/.instance_id)"

#!/bin/bash
# Create a Cloud TPU VM.
#
#   PROJECT=my-gcp-project ZONE=us-east5-b ACCEL=v6e-1 bash torchtpu/scripts/launch_tpu_vm.sh
#
# Phase 0 / 1: a single-chip slice (v6e-1: 32 GB HBM, or v5litepod-1: 16 GB HBM).
# Phase 2a:    a single v6e chip may fit Qwen3.5-4B + LoRA in bf16; otherwise v6e-4.
#
# Check that the TorchTPU docs list your TPU generation and runtime version
# before launching: TorchTPU is new, and supported generations may be narrower
# than PyTorch/XLA's.
set -euo pipefail

: "${PROJECT:?set PROJECT}"
ZONE=${ZONE:-us-east5-b}
ACCEL=${ACCEL:-v6e-1}
NAME=${NAME:-torchtpu-grpo-bench}
# Runtime image per generation (verify against current Cloud TPU docs).
case "$ACCEL" in
  v6e-*)      VERSION=${VERSION:-v2-alpha-tpuv6e} ;;
  v5litepod-*) VERSION=${VERSION:-v2-alpha-tpuv5-lite} ;;
  *)          : "${VERSION:?set VERSION (runtime image) for $ACCEL}" ;;
esac

gcloud compute tpus tpu-vm create "$NAME" \
  --project "$PROJECT" --zone "$ZONE" \
  --accelerator-type "$ACCEL" --version "$VERSION" \
  ${SPOT:+--spot}

echo "ssh:    gcloud compute tpus tpu-vm ssh $NAME --project $PROJECT --zone $ZONE"
echo "delete: gcloud compute tpus tpu-vm delete $NAME --project $PROJECT --zone $ZONE"

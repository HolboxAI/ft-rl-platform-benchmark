#!/bin/bash
# Phase 0 on TPU: CholecT50 SFT checkpoint + one val frame at the run's shapes.
# TorchTPU by default; TPU_PATH=torchxla for the PyTorch/XLA baseline. Needs common/fetch_cholect50.sh first.
set -uo pipefail
cd "$(dirname "$0")/../.."
source torchtpu/scripts/tpu_env.sh
python -m common.smoke_test --backend "$BACKEND" --model "${SFT_CKPT:-$HOME/models/cholec-sft-qwen3.5-4b}" \
  --out "torchtpu/logs/smoke_${TPU_PATH}.json" "$@" 2>&1 | tee "torchtpu/logs/smoke_${TPU_PATH}_console.log"

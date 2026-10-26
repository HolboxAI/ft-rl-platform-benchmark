#!/usr/bin/env bash
# Single-chip smoke test of the verl CholecT50 LoRA GRPO run on TPU.
#
# Isolates the training loop (rollout -> reward -> GRPO update) on one chip
# before tackling the multi-chip topology. Run on the VM.
set -euo pipefail
export VERL_PLATFORM=tpu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export USE_HUB_KERNELS=NO
export TOKENIZERS_PARALLELISM=false

# Single-chip TorchTPU topology (chip 0, one process).
export TORCH_TPU_TOPOLOGY="1,1,1"
export TORCH_TPU_SLICEBUILDER_ADDRESSES="localhost:9470"
export TPU_VISIBLE_DEVICES="0"
export TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1"
export TPU_PROCESS_BOUNDS="1,1,1"
export TPU_PROCESS_ADDRESSES="localhost:9478"
export TPU_VISIBLE_CHIPS="0"
export TPU_PROCESS_PORT="9478"
export CLOUD_TPU_TASK_ID="0"

cd ~/verl-tpu
bash scripts/train.sh trainer.n_gpus_per_node=1 trainer.total_training_steps=1 "$@"

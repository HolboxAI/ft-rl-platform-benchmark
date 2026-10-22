#!/bin/bash
# GPU control: the SAME SkyRL Tinker API + the SAME CholecT50 client, on the FSDP
# backend (vLLM rollout), where vision is supported. This validates the client
# and gives the number the TPU / Trainium runs should be compared against.
#
# Hardware: g5.12xlarge (4x A10G), as the verl / SkyRL / NeMo-RL GPU legs.
#   MODEL=$HOME/models/cholec-sft-qwen3.5-4b bash skyrl/gpu_control/start_server.sh
#
# SkyRL's docs note VLM training needs a newer vLLM than the repo pins; see
# docs.skyrl.ai "Vision-Language RL" -> Requirements before starting.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SKYRL_DIR=${SKYRL_DIR:-$HOME/SkyRL}
MODEL=${MODEL:?set MODEL}
NUM_GPUS=${NUM_GPUS:-4}
export PATH="$HOME/.local/bin:$PATH"
# Same Qwen3.5 kernel workaround as the SkyRL GPU leg (hub kernels emit garbage on A10G).
export USE_HUB_KERNELS=NO

cd "$SKYRL_DIR"
uv run --isolated --extra tinker --extra fsdp -m skyrl.tinker.api \
  --base-model "$MODEL" --backend fsdp \
  --backend-config "{\"trainer.remove_microbatch_padding\": false,
                     \"trainer.placement.policy_num_gpus_per_node\": $NUM_GPUS,
                     \"generator.inference_engine.num_engines\": $NUM_GPUS,
                     \"trainer.policy.fsdp_config.cpu_offload\": true}" \
  --port "${PORT:-8000}" \
  2>&1 | tee "$REPO/skyrl/logs/server_gpu_$(date -u +%Y%m%d-%H%M%S).log"

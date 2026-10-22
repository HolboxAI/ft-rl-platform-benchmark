#!/bin/bash
# Phase 0 on Trainium: CholecT50 SFT checkpoint + one val frame at the run's shapes:
# device, forward, LoRA train step, sampling. Mostly compile time. Needs common/fetch_cholect50.sh first.
set -uo pipefail
cd "$(dirname "$0")/../.."
source trainium/scripts/neuron_env.sh
python -m common.smoke_test --backend neuron --model "${SFT_CKPT:-$HOME/models/cholec-sft-qwen3.5-4b}" \
  --out trainium/logs/smoke.json "$@" 2>&1 | tee trainium/logs/smoke_console.log

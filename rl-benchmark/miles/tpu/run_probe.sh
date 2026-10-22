#!/bin/bash
# Miles on Cloud TPU: install into the TorchTPU venv (from torchtpu/scripts/setup_env.sh)
# and run the stage probe. Miles' TPU hook targets TPU v7x; on v6e/v5e a pass at the
# platform stage still shows the hook works, but record the generation used.
set -uo pipefail
cd "$(dirname "$0")/../.."
MILES_DIR=${MILES_DIR:-$HOME/miles}
MILES_REF=${MILES_REF:-8e99947}   # commit audited in miles/metrics/porting_audit.json (2026-10-22)
[ -d "$MILES_DIR/.git" ] || git clone https://github.com/radixark/miles "$MILES_DIR"
git -C "$MILES_DIR" fetch --depth 1 origin "$MILES_REF" 2>/dev/null; git -C "$MILES_DIR" checkout "$MILES_REF"

source .venv-torchtpu/bin/activate
# --no-deps: Miles' requirements pull CUDA-only wheels (sglang kernels, ring_flash_attn,
# nvidia-resiliency-ext). Install only the light, pure-Python ones and let the probe
# report what else is missing.
pip install --no-deps -e "$MILES_DIR" 2>&1 | tail -1
pip install omegaconf pybase64 orjson httpx "ray[default]>=2.56,<2.59" qwen_vl_utils 2>&1 | tail -1
# Rollout engine: try SGLang as-is and record the outcome (expected to fail on TPU).
pip install "sglang" 2>&1 | tail -2 | tee miles/logs/tpu_sglang_install.txt

python miles/scripts/probe_miles.py --backend tpu --miles-src "$MILES_DIR" --out miles/logs/probe_tpu.json

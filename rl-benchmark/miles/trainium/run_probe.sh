#!/bin/bash
# Miles on Trainium: install into the Neuron venv and run the stage probe.
# Miles has no Neuron platform, so the probe passes MILES_ACCELERATOR=neuron and
# expects a ValueError at the platform stage. The import_surface stage then shows
# how much of Miles loads at all next to torch-neuronx.
set -uo pipefail
cd "$(dirname "$0")/../.."
source trainium/scripts/neuron_env.sh
MILES_DIR=${MILES_DIR:-$HOME/miles}
MILES_REF=${MILES_REF:-8e99947}
[ -d "$MILES_DIR/.git" ] || git clone https://github.com/radixark/miles "$MILES_DIR"
git -C "$MILES_DIR" fetch --depth 1 origin "$MILES_REF" 2>/dev/null; git -C "$MILES_DIR" checkout "$MILES_REF"

# Never let pip touch torch / torch-xla / torch-neuronx in the Neuron venv.
pip install --no-deps -e "$MILES_DIR" 2>&1 | tail -1
pip install --upgrade-strategy only-if-needed omegaconf pybase64 orjson httpx "ray[default]>=2.56,<2.59" qwen_vl_utils 2>&1 | tail -1
pip download --no-deps -d /tmp/sglang-probe sglang 2>&1 | tail -1 | tee miles/logs/trainium_sglang_download.txt

python miles/scripts/probe_miles.py --backend neuron --miles-src "$MILES_DIR" --out miles/logs/probe_trainium.json

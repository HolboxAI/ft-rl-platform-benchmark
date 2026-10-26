#!/usr/bin/env bash
# Install verl + the TorchNeuron platform plugin on a Trainium2 instance.
#
# Verified 2026-10-26 on "Deep Learning AMI Neuron (Ubuntu 26.04) SDK 2.33.0"
# (ami-0f7a2cecca50fc858), venv /opt/aws_neuronx_venv_torch_neuronx_2_14_0,
# Python 3.14.4. Gets verl 0.9.0 to import and the `neuron` platform to register
# (gate_test.py passes). Idempotent.
#
# Why not `pip install verl` plain:
#   * verl 0.9.1 requires Python <3.13 (DLAMI has 3.14)  -> use 0.9.0 instead
#     (0.9.0 already ships verl/plugin/platform and pins Python >=3.10 only).
#   * verl pins tensordict<=0.10.0 (no 3.14 wheels)        -> override 0.15.3.
#   * verl pins transformers<5.11 (qwen3_5 needs 5.18.0)   -> keep 5.18.0.
#   * verl imports orjson but does not declare it          -> install manually.
set -euo pipefail

VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
NEURON_PIP="https://pip.repos.neuron.amazonaws.com"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "== 0. base =="
python --version
python -c "import torch; print('torch', torch.__version__)"

echo "== 1. pin torch-neuronx 2.14.3.0.282 (AMI bundles .1, crashes on repeat_interleave) =="
python -m pip install -q -U pip
python -m pip install -q --extra-index-url "$NEURON_PIP" "torch-neuronx==2.14.3.0.282+dc70ffa" --no-deps

echo "== 2. transformers 5.18.0 (qwen3_5 arch) + vision/data =="
python -m pip install -q "transformers==5.18.0" accelerate pillow safetensors datasets
python -m pip install -q "torchvision==0.29.1" --no-deps
python -m pip install -q "peft==0.21.2" --no-deps

echo "== 3. verl 0.9.0 core (--no-deps skips its bad pins) =="
python -m pip install -q "verl==0.9.0" --no-deps

echo "== 4. verl runtime deps (ray/hydra/data/logging; the two pin overrides below) =="
python -m pip install -q \
  "ray[default]>=2.41.0" hydra-core torchdata codetiming pybind11 pylatexenc \
  dill pandas "pyarrow>=19.0.0" wandb tensorboard orjson msgpack aiohttp \
  "numpy>=2.0.0" packaging

echo "== 5. override verl's incompatible pins =="
python -m pip install -q "tensordict==0.15.3"      # verl wants <=0.10.0 (no 3.14 wheels)
# transformers stays 5.18.0 (overrides verl's <5.11)

echo "== 6. platform plugin (registers `neuron` via verl.plugins) =="
python -m pip install -q --no-deps -e "$ROOT/plugin"

echo "== 7. verify =="
python - <<'PY'
import torch, torch_neuronx
print("torch", torch.__version__)
print("torch_neuronx", torch_neuronx.__version__)
print("neuron device:", torch.device("neuron"))
import verl
from verl.plugin.platform import PlatformRegistry
names = PlatformRegistry.registered_names()
print("REGISTERED_PLATFORMS", names)
assert "neuron" in names, f"neuron not registered; got {names}"
print("NEURON_PLATFORM_OK")
PY

echo "DONE setup"

#!/bin/bash
# Setup for the Ubuntu 26.04 "torch-neuronx 2.14.0" DLAMI.
# TorchNeuron is already preinstalled in /opt/aws_neuronx_venv_torch_neuronx_2_14_0.
# We only add transformers (4.57.x for qwen3_5), trl, accelerate, pillow.
set -euo pipefail

VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "=== python / torch ==="
python --version
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_neuronx; print('torch_neuronx', torch_neuronx.__version__)"
python -c "import torch; print('neuron device:', torch.device('neuron'))"

echo "=== install trl / transformers / accelerate / pillow ==="
python -m pip install -q -U pip
# transformers 5.x is required for the qwen3_5 architecture (4.x does not have it).
python -m pip install -q "transformers==5.18.0" "trl==1.16.0" accelerate pillow safetensors

echo "=== verify ==="
python - <<'PY'
import transformers, trl, accelerate
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("accelerate", accelerate.__version__)

# qwen3_5 support check
import importlib
m = importlib.import_module("transformers.models.qwen3_5")
print("qwen3_5 has Qwen3_5ForConditionalGeneration:",
      hasattr(m, "Qwen3_5ForConditionalGeneration"))
print("qwen3_5 has Qwen3_5Config:", hasattr(m, "Qwen3_5Config"))

from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer
print("core imports OK")
PY

echo "DONE setup2"

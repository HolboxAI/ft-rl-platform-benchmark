#!/bin/bash
# Complete setup for the "torch-neuronx 2.14.0" DLAMI (Ubuntu 26.04).
# TorchNeuron is preinstalled in /opt/aws_neuronx_venv_torch_neuronx_2_14_0.
# The AMI bundles torch-neuronx 2.14.3.0.1, which crashes on torch.repeat_interleave
# (see TRAINIUM_ISSUES_LOG.md #1). Pin 2.14.3.0.282 (--no-deps), then add the rest.
set -euo pipefail

VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "=== python / torch ==="
python --version
python -c "import torch; print('torch', torch.__version__)"

echo "=== torch-neuronx BEFORE pin ==="
python -c "import torch_neuronx; print('torch_neuronx', torch_neuronx.__version__)" 2>&1 | head -1

echo "=== pin torch-neuronx 2.14.3.0.282 (--no-deps) ==="
python -m pip install -q --extra-index-url https://pip.repos.neuron.amazonaws.com "torch-neuronx==2.14.3.0.282+dc70ffa" --no-deps

echo "=== install trl / transformers / accelerate / peft / datasets / pillow / torchvision ==="
python -m pip install -q -U pip
python -m pip install -q "transformers==5.18.0" "trl==1.16.0" accelerate pillow safetensors datasets
python -m pip install -q "peft==0.21.2" --no-deps
# Qwen3VLVideoProcessor (part of the qwen3_5 AutoProcessor) imports torchvision.
# --no-deps keeps pip from touching the Neuron torch (2.14.0+cu130).
python -m pip install -q "torchvision==0.29.1" --no-deps

echo "=== verify ==="
python - <<'PY'
import torch
import torch_neuronx
print("torch", torch.__version__)
print("torch_neuronx", torch_neuronx.__version__)
print("neuron device:", torch.device("neuron"))
import transformers, trl, accelerate, peft, datasets
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("accelerate", accelerate.__version__)
print("peft", peft.__version__)
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers import AutoModelForImageTextToText
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
print("core imports OK")
PY

echo "DONE setup"

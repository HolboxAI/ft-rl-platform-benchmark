#!/bin/bash
# Set up TorchNeuron + TRL on a Trainium2 instance (trn2.48xlarge).
# Run once as the ubuntu user. Idempotent enough to re-run.
set -euo pipefail

export NEURON_PIP="https://pip.repos.neuron.amazonaws.com"
VENV=/home/ubuntu/venv

echo "=== python ==="
python3 --version

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -q -U pip

echo "=== torch-neuronx (TorchNeuron native backend) ==="
python -m pip install -q --extra-index-url "$NEURON_PIP" "torch-neuronx==2.14.3.0.281+dc70ffa"

echo "=== transformers / accelerate / trl / datasets ==="
python -m pip install -q transformers accelerate trl datasets pillow

echo "=== verify ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
import torch_neuronx  # noqa: F401  (registers the `neuron` device)
try:
    print("neuron device registered:", torch.device("neuron"))
except Exception as e:
    print("neuron device ERROR:", e)
import transformers, trl, accelerate
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("accelerate", accelerate.__version__)
PY

echo "=== neuron devices ==="
ls -l /dev/neuron* 2>&1 | head
which neuron-ls 2>&1 || true
neuron-ls 2>&1 | head -40 || true

echo "=== DONE setup ==="

#!/usr/bin/env bash
# One-time setup on trn1 / trn2 / trn3 (AWS Deep Learning AMI Neuron, Ubuntu 22.04).
# Installs TRL into the AMI's torch-neuronx venv WITHOUT replacing its torch / torch-xla.
set -euo pipefail
cd "$(dirname "$0")/.."
NEURON_VENV="${NEURON_VENV:-$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | grep -v inference | head -1)}"
[ -n "$NEURON_VENV" ] || { echo "no /opt/aws_neuronx_venv_pytorch_* found; set NEURON_VENV"; exit 1; }
# shellcheck disable=SC1091
source "$NEURON_VENV/bin/activate"
# constraint file pins the Neuron-provided torch stack so pip can't upgrade it
pip list --format=freeze | grep -i -E '^(torch|torch-xla|torch-neuronx|torchvision|neuronx-cc|libneuronxla)==' > /tmp/neuron_pins.txt
pip install -q -r requirements.txt -c /tmp/neuron_pins.txt
bash scripts/setup_reward_env.sh
python -c "import torch, torch_neuronx, trl; print('torch', torch.__version__, 'torch-neuronx', torch_neuronx.__version__, 'trl', trl.__version__)"
neuron-ls || true

#!/bin/bash
# One-time setup on the trn instance (run from the repo root after copying it over).
#
#   bash trainium/scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source trainium/scripts/neuron_env.sh

# Record exactly what the DLAMI shipped, before we add anything.
mkdir -p trainium/logs
{
  echo "# $(date -u +%FT%TZ)  $(hostname)  ${INSTANCE_TYPE}"
  neuron-ls 2>&1 || true
  pip list 2>/dev/null | grep -iE "neuron|torch|xla|transformers|jax" || true
  apt list --installed 2>/dev/null | grep -i neuron || true
} | tee trainium/logs/environment.txt

# transformers / peft on top of the Neuron venv. Do NOT upgrade torch or
# torch-xla: they must stay at the versions torch-neuronx was built against.
pip install --upgrade-strategy only-if-needed -r requirements-common.txt

python -c "import torch, torch_neuronx, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

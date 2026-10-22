#!/bin/bash
# One-time setup on the TPU VM (run from the repo root after copying it over).
#
# Creates two venvs so the same VM can run both TPU paths:
#   .venv-torchtpu  — TorchTPU (native "tpu" device)        -> --backend tpu
#   .venv-torchxla  — PyTorch/XLA (previous TPU path)        -> --backend tpu-xla
#
#   bash torchtpu/scripts/setup_env.sh            # both
#   WITH_XLA=0 bash torchtpu/scripts/setup_env.sh # TorchTPU only
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p torchtpu/logs

python3 -m venv .venv-torchtpu
# shellcheck disable=SC1091
source .venv-torchtpu/bin/activate
pip install --upgrade pip
# Install line from the Cloud TPU "PyTorch on TPU" docs. If the docs give a
# specific index URL or pinned torch version for your runtime, use that instead
# and record it in config/TORCHTPU_CONFIG.md.
pip install ${TORCH_TPU_PIP:-torch_tpu}
pip install -r requirements-common.txt
python -c "import torch, torch_tpu; t = torch.ones(2, device='tpu'); print('torch', torch.__version__, 'tpu ok', (t * 2).sum().item())"
deactivate

if [ "${WITH_XLA:-1}" = "1" ]; then
  python3 -m venv .venv-torchxla
  # shellcheck disable=SC1091
  source .venv-torchxla/bin/activate
  pip install --upgrade pip
  pip install torch 'torch_xla[tpu]'
  pip install -r requirements-common.txt
  PJRT_DEVICE=TPU python -c "import torch_xla.core.xla_model as xm; print('xla device', xm.xla_device())"
  deactivate
fi

{
  echo "# $(date -u +%FT%TZ)  $(hostname)"
  curl -s -m 2 -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type || true
  echo
  for v in .venv-torchtpu .venv-torchxla; do
    [ -d "$v" ] && { echo "## $v"; "$v/bin/pip" list 2>/dev/null | grep -iE "torch|xla|jax|libtpu|transformers|peft"; }
  done
} | tee torchtpu/logs/environment.txt

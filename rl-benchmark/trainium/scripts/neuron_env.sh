#!/bin/bash
# Source this on the trn instance: activates the Neuron PyTorch venv and sets
# the compiler/runtime env shared by every Trainium run.
#
#   source trainium/scripts/neuron_env.sh

# The Neuron DLAMI ships a pre-built venv: /opt/aws_neuronx_venv_pytorch_<ver> on older
# images, /opt/aws_neuronx_venv_torch_neuronx_<ver> on the torch-neuronx 2.14 image.
# Pick the newest unless NEURON_VENV is set explicitly.
if [ -z "${NEURON_VENV:-}" ]; then
  NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_torch_neuronx_* /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | sort -V | tail -1)
fi
export PATH="/opt/aws/neuron/bin:$PATH"   # neuron-ls, neuron-top
if [ -n "${NEURON_VENV}" ] && [ -f "${NEURON_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${NEURON_VENV}/bin/activate"
else
  echo "WARN: no Neuron venv found under /opt; using current python ($(command -v python))" >&2
fi

# Compiler: transformer-tuned scheduling; cache compiled graphs (NEFFs) on local
# disk so re-runs and later steps skip the compile.
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---model-type=transformer --retry_failed_compilation}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/var/tmp/neuron-compile-cache}"
# Single-device runs use one NeuronCore. trn1.2xlarge has 2 (1 chip).
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-1}"
export MALLOC_ARENA_MAX=64
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export INSTANCE_TYPE="${INSTANCE_TYPE:-$(curl -s -m 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo unknown)}"

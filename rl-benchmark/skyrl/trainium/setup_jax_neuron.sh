#!/bin/bash
# Make SkyRL's JAX backend see Trainium through the JAX NeuronX plugin (beta).
# Run on a trn instance (Neuron DLAMI) after skyrl/scripts/setup_skyrl.sh.
#
# Approach: build SkyRL's environment with the plain `jax` extra (no CUDA/TPU
# jaxlib), then add libneuronxla + jax-neuronx from the Neuron pip index, and
# start the server with `uv run --no-sync` so uv does not remove them again.
# The first likely failure is a version conflict: SkyRL requires jax>=0.8 and
# jax-neuronx pins a tested jax/jaxlib. Record what pip resolves.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SKYRL_DIR=${SKYRL_DIR:-$HOME/SkyRL}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$REPO/skyrl/logs"
LOG="$REPO/skyrl/logs/trainium_jax_setup.txt"

cd "$SKYRL_DIR"
uv sync --extra jax --extra tinker 2>&1 | tail -3 | tee "$LOG"
# Neuron's pip index (see the Neuron JAX setup guide for the current command).
uv pip install --extra-index-url https://pip.repos.neuron.amazonaws.com \
  "${JAX_NEURON_PIN:-jax-neuronx[stable]}" 2>&1 | tee -a "$LOG"
uv run --no-sync python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__); print(jax.devices())" \
  2>&1 | tee -a "$LOG"

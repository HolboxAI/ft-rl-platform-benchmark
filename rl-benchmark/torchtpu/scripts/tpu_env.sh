#!/bin/bash
# Source this on the TPU VM. Picks the venv for the requested path.
#
#   TPU_PATH=torchtpu source torchtpu/scripts/tpu_env.sh   # default, --backend tpu
#   TPU_PATH=torchxla source torchtpu/scripts/tpu_env.sh   # baseline, --backend tpu-xla
TPU_PATH=${TPU_PATH:-torchtpu}
# shellcheck disable=SC1090
source ".venv-${TPU_PATH}/bin/activate"
if [ "$TPU_PATH" = "torchtpu" ]; then
  export BACKEND=tpu
else
  export BACKEND=tpu-xla
  export PJRT_DEVICE=TPU
  # Persistent compile cache for PyTorch/XLA so warm reruns skip compiles.
  export XLA_PERSISTENT_CACHE_PATH=${XLA_PERSISTENT_CACHE_PATH:-/tmp/xla_cache}
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export INSTANCE_TYPE="${INSTANCE_TYPE:-$(curl -s -m 2 -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type 2>/dev/null || echo unknown)}"

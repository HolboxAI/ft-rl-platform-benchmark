#!/usr/bin/env bash
# One-time setup on the ROLLOUT VM. Builds the venv for one engine:
#
#   ENGINE=vllm bash ~/tpu_rl_benchmark/setup/setup_rollout_vm.sh   # vLLM on TPU (tpu-inference)
#   ENGINE=hf   bash ~/tpu_rl_benchmark/setup/setup_rollout_vm.sh   # transformers generate on TorchTPU
#
# Both venvs get the server's own deps (fastapi, uvicorn, transformers with
# Qwen3.5 support for rendering). Run both if you want to compare engines.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE=${ENGINE:-vllm}
VENV=${VENV:-$HOME/.venv-rollout-$ENGINE}
VLLM_TPU_SPEC=${VLLM_TPU_SPEC:-vllm-tpu}
TORCH_TPU_SPEC=${TORCH_TPU_SPEC:-torch_tpu==0.2.0}
# SkyRL 2c54cefc pins transformers <=5.16.1; render with the same version the trainer uses.
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-transformers==5.16.1}
LOG="$ROOT/logs/setup_rollout_${ENGINE}_$(date -u +%Y%m%d-%H%M%S).txt"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear --python 3.12 "$VENV"
PY="$VENV/bin/python"
SERVER_DEPS=(fastapi "uvicorn[standard]" safetensors pillow requests pyarrow)

case "$ENGINE" in
  vllm)
    # vLLM's TPU build (tpu-inference plugin). It pins its own torch/jax/transformers.
    uv pip install --python "$PY" "$VLLM_TPU_SPEC" "${SERVER_DEPS[@]}"
    TORCH_VER=$("$PY" -c "import torch; print(torch.__version__.split('+')[0])")
    uv pip install --python "$PY" "torch==$TORCH_VER" torchvision --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    # Qwen3.5 needs a recent transformers for the processor. If vllm-tpu pins an older one,
    # this upgrade may break it: the import check below says which.
    "$PY" -c "from transformers.models import qwen3_5" 2>/dev/null \
      || uv pip install --python "$PY" "$TRANSFORMERS_SPEC"
    ;;
  hf)
    uv pip install --python "$PY" "$TORCH_TPU_SPEC" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    TORCH_VER=$("$PY" -c "import torch; print(torch.__version__.split('+')[0])")
    uv pip install --python "$PY" "torch==$TORCH_VER" torchvision "$TRANSFORMERS_SPEC" accelerate "${SERVER_DEPS[@]}" \
      --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    ;;
  *) echo "ENGINE must be vllm or hf"; exit 1 ;;
esac

echo "== versions"
uv pip list --python "$PY" 2>/dev/null | grep -iE "^(vllm|tpu.inference|torch|jax|libtpu|transformers)" || true
echo "== imports"
"$PY" - <<EOF
import sys
sys.path.insert(0, "$ROOT/rollout")
import transformers
from transformers.models import qwen3_5  # noqa: F401
import torchvision  # noqa: F401  (Qwen3.5 processor)
print("ok transformers", transformers.__version__, "knows qwen3_5; torchvision", torchvision.__version__)
import mm, engines  # noqa: F401
print("ok rollout modules")
if "$ENGINE" == "vllm":
    import vllm
    print("ok vllm", vllm.__version__)
    try:
        import tpu_inference
        print("ok tpu_inference", getattr(tpu_inference, "__version__", "?"))
    except ImportError as e:
        print("WARN no tpu_inference:", e)
else:
    import torch, torch_tpu  # noqa: F401
    print("ok tpu:", (torch.ones(2, device="tpu") * 2).sum().item())
EOF
echo "setup done; log: $LOG"

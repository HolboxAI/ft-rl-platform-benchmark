#!/usr/bin/env bash
# One-time setup on the TRAINER VM: TorchTPU + verl (v0.9.1) in one venv.
#
#   bash ~/verl-tpu/setup/setup_trainer_vm.sh
#
# verl's own install pulls vLLM / flash-attn / flashinfer / sglang / cupy and a
# CUDA torch, none of which run on TPU. Instead:
#   1. torch_tpu (TorchTPU) with the CPU torch it was built against,
#   2. verl's `verl-core` deps minus the CUDA-only ones, with torch pinned
#      (constraints) so no dependency swaps in a CUDA torch,
#   3. verl itself `--no-deps` (checked out at v0.9.1),
#   4. the three Qwen3.5 patches (verl bugs, hardware-independent),
#   5. import checks of everything the trainer process loads.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR=${VERL_DIR:-$HOME/verl}                 # verl @ v0.9.1
VENV=${VERL_VENV:-$HOME/.venv-verl-tpu}
TORCH_TPU_SPEC=${TORCH_TPU_SPEC:-torch_tpu==0.2.0}
# verl-core deps that are CUDA-only or otherwise unsupported on TPU.
EXCLUDE=${EXCLUDE:-"nvtx torchcodec flash-attn cupy-cuda13x liger-kernel vllm sglang triton trl flash-linear-attention TransferQueue"}
LOG="$ROOT/logs/setup_trainer_$(date -u +%Y%m%d-%H%M%S).txt"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# verl checkout at v0.9.1 (shallow).
if [ ! -f "$VERL_DIR/pyproject.toml" ]; then
  git clone --depth 1 --branch v0.9.1 https://github.com/volcengine/verl.git "$VERL_DIR"
fi

uv venv --clear --python 3.12 "$VENV"
PY="$VENV/bin/python"

# 1. TorchTPU. The extra index keeps pip from resolving a CUDA torch.
uv pip install --python "$PY" "$TORCH_TPU_SPEC" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
TORCH_VER=$("$PY" -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch $TORCH_VER from torch_tpu"
CONSTRAINTS="$ROOT/logs/verl_constraints.txt"
echo "torch==$TORCH_VER" > "$CONSTRAINTS"

# 2. verl-core deps minus CUDA-only packages.
REQS="$ROOT/logs/verl_requirements.txt"
"$PY" -I - "$VERL_DIR/pyproject.toml" "$EXCLUDE" > "$REQS" <<'EOF'
import re, sys, tomllib
proj = tomllib.load(open(sys.argv[1], "rb"))["project"]
exclude = set(x.lower() for x in sys.argv[2].split())
out, seen = [], set()
for r in proj["optional-dependencies"]["verl-core"]:
    name = re.split(r"[\s<>=!~;\[@]", r, 1)[0].lower()
    if name == "verl" or name in exclude or r in seen:
        continue
    seen.add(r)
    out.append(r)
print("\n".join(out))
EOF
echo "installing $(wc -l < "$REQS") verl-core requirements (excluded: $EXCLUDE)"
uv pip install --python "$PY" -r "$REQS" -c "$CONSTRAINTS" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# 3. verl itself, no deps.
uv pip install --python "$PY" --no-deps -e "$VERL_DIR"

# 4. torchvision (CPU build matching torch) + runtime helpers.
uv pip install --python "$PY" -c "$CONSTRAINTS" torchvision pyarrow requests --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# 5. The three Qwen3.5 patches (verl bugs: mRoPE position_ids + tensordict).
"$PY" "$ROOT/patches/patch_qwen35.py"     "$VERL_DIR/verl/models/transformers/qwen3_5.py"
"$PY" "$ROOT/patches/patch_tensordict.py" "$VERL_DIR/verl/utils/tensordict_utils.py"
"$PY" "$ROOT/patches/patch_fix.py"        "$VERL_DIR/verl/utils/tensordict_utils.py"

# 6. Checks.
echo "== versions"
uv pip list --python "$PY" 2>/dev/null | grep -iE "^(verl|torch|transformers|ray|peft|accelerate|tensordict|vllm|flash)" || true
echo "== device"
VERL_PLATFORM=tpu "$PY" - <<'EOF'
import torch
import torch_tpu  # noqa: F401
from torch_tpu._loader import load
load()
x = torch.ones(4, device="tpu") * 2
print("tpu ok:", x.sum().item(), "| accelerator available:", torch.accelerator.is_available())
EOF
echo "== imports (trainer process)"
(cd "$VERL_DIR" && VERL_PLATFORM=tpu "$PY" - <<'EOF'
import importlib
for m in [
    "verl",
    "verl.trainer.main_ppo",
    "verl.workers.engine.fsdp.transformer_impl",
    "verl.workers.rollout.hf_rollout",
    "verl.plugin.platform",
]:
    importlib.import_module(m)
    print("ok", m)
from verl.plugin.platform import get_platform
p = get_platform()
print("platform:", p.device_name, p.vendor_name, p.communication_backend_name())
try:
    import vllm  # noqa: F401
    print("NOTE: vllm is importable in the trainer venv")
except ImportError:
    print("ok   vllm absent (expected)")
EOF
)
echo "setup done; log: $LOG"

#!/usr/bin/env bash
# One-time setup on the TRAINER VM: TorchTPU + SkyRL (skyrl-tpu patch) in one venv.
#
#   bash ~/tpu_rl_benchmark/skyrl-tpu/setup/setup_trainer_vm.sh
#
# SkyRL's own install (`uv sync --extra fsdp`) pulls vLLM 0.31, flash-attn, flashinfer,
# nixl and a CUDA torch, none of which run on TPU. Instead:
#   1. torch_tpu (TorchTPU) with the CPU torch it was built against,
#   2. SkyRL --no-deps, plus its base + `skyrl-train` deps minus the CUDA-only ones,
#      with torch pinned (constraints) so no dependency swaps in a CUDA torch,
#   3. an import check of everything the trainer process loads.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKYRL_TPU_DIR=${SKYRL_TPU_DIR:-$HOME/SkyRL-tpu}
VENV=${SKYRL_VENV:-$HOME/.venv-skyrl-tpu}
TORCH_TPU_SPEC=${TORCH_TPU_SPEC:-torch_tpu==0.2.0}
EXCLUDE=${EXCLUDE:-"flash-attn nixl s5cmd skyrl-gym debugpy vllm flashinfer-python"}
LOG="$ROOT/logs/setup_trainer_$(date -u +%Y%m%d-%H%M%S).txt"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -f "$SKYRL_TPU_DIR/skyrl/backends/skyrl_train/utils/device.py" ] || {
  echo "$SKYRL_TPU_DIR has no skyrl-tpu patch; run 'bash gcloud/vm.sh sync' from the laptop"; exit 1; }

uv venv --clear --python 3.12 "$VENV"
PY="$VENV/bin/python"

# 1. TorchTPU. The extra index keeps pip from resolving a CUDA torch.
uv pip install --python "$PY" "$TORCH_TPU_SPEC" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
TORCH_VER=$("$PY" -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch $TORCH_VER from torch_tpu"
CONSTRAINTS="$ROOT/logs/trainer_constraints.txt"
echo "torch==$TORCH_VER" > "$CONSTRAINTS"

# 2. SkyRL deps from its pyproject, minus CUDA-only packages.
REQS="$ROOT/logs/trainer_requirements.txt"
"$PY" -I - "$SKYRL_TPU_DIR/pyproject.toml" "$EXCLUDE" > "$REQS" <<'EOF'
import re, sys, tomllib
proj = tomllib.load(open(sys.argv[1], "rb"))["project"]
exclude = set(sys.argv[2].split())
for r in proj["dependencies"] + proj["optional-dependencies"]["skyrl-train"]:
    name = re.split(r"[\s<>=!~;\[]", r, 1)[0].lower()
    if name not in exclude and not name.startswith("skyrl"):
        print(r)
EOF
echo "installing $(wc -l < "$REQS") SkyRL requirements (excluded: $EXCLUDE)"
uv pip install --python "$PY" -r "$REQS" -c "$CONSTRAINTS" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install --python "$PY" -c "$CONSTRAINTS" -e "$SKYRL_TPU_DIR/skyrl-gym"
uv pip install --python "$PY" --no-deps -e "$SKYRL_TPU_DIR"
# data.py (verl parquet -> SkyRL parquet) and the probes. torchvision (CPU build matching
# torch): Qwen3.5's AutoProcessor loads a video processor that requires it.
uv pip install --python "$PY" -c "$CONSTRAINTS" pyarrow requests torchvision --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# 3. Checks.
echo "== versions"
uv pip list --python "$PY" 2>/dev/null | grep -iE "^(skyrl|torch|transformers|ray|peft|accelerate|vllm|flash)" || true
echo "== device"
SKYRL_DEVICE=tpu "$PY" - <<'EOF'
import torch
import torch_tpu  # noqa: F401
x = torch.ones(4, device="tpu") * 2
print("tpu ok:", x.sum().item(), "| devices visible to this process:", torch.accelerator.device_count())
EOF
echo "== imports (trainer process)"
(cd "$SKYRL_TPU_DIR" && SKYRL_DEVICE=tpu PYTHONPATH="$ROOT/skyrl_cholect50" "$PY" - <<'EOF'
import importlib
for m in [
    "skyrl.backends.skyrl_train.utils.device",
    "skyrl.backends.skyrl_train.weight_sync.reload",
    "skyrl.backends.skyrl_train.workers.model_wrapper",
    "skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker",
    "skyrl.train.entrypoints.main_base",
    "skyrl.train.generators.skyrl_vlm_generator",
    "cholec_env",
]:
    importlib.import_module(m)
    print("ok", m)
try:
    import vllm  # noqa: F401
    print("NOTE: vllm is importable in the trainer venv")
except ImportError:
    print("ok   vllm absent (expected)")
EOF
)
echo "setup done; log: $LOG"

#!/bin/bash
# TitanRL (torchtitan/rl) on Trainium2 — compatibility probe.
#
# Captures the exact CUDA-wall failure modes WITHOUT touching the DLAMI neuron
# torch (the TitanRL install probes run in an isolated venv). Writes a full log
# and prints TITANRL_PROBE_COMPLETE when done.
set -uo pipefail

LOG=~/titanrl_probe.log
exec > >(tee -a "$LOG") 2>&1

echo "=== [0] start $(date -u) ==="
echo "hostname: $(hostname)"
echo "--- neuron devices ---"
ls -l /dev/neuron* 2>&1 | head
neuron-ls 2>&1 | head -20 || true

echo ""
echo "=== [1] neuron environment reality (DLAMI torch-neuronx venv) ==="
# shellcheck disable=SC1091
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.hip:", getattr(torch.version, "hip", None))
try:
    d = torch.device("neuron")
    print("torch.device('neuron'): OK ->", d)
except Exception as e:
    print("torch.device('neuron') FAILED:", type(e).__name__, str(e))
try:
    import torch_neuronx
    print("torch_neuronx:", torch_neuronx.__version__)
except Exception as e:
    print("torch_neuronx import FAILED:", type(e).__name__, str(e))
PY

echo ""
echo "=== [2] clone torchtitan ==="
cd "$HOME"
if [ ! -d "$HOME/torchtitan/.git" ]; then
  git clone --depth 1 https://github.com/pytorch/torchtitan.git "$HOME/torchtitan" 2>&1 | tail -3
fi
cd "$HOME/torchtitan"
echo "torchtitan rev: $(git rev-parse HEAD 2>/dev/null)"

echo ""
echo "=== [3] has_cuda_capability gate (vllm/engine.py:283 / tools/utils.py:25) ==="
cd "$HOME/torchtitan"
python - <<'PY'
import sys; sys.path.insert(0, ".")
import torch
from torchtitan.tools.utils import has_cuda_capability
print("torch.cuda.is_available():", torch.cuda.is_available())
print("has_cuda_capability(9, 0):", has_cuda_capability(9, 0))
print("has_cuda_capability(8, 0):", has_cuda_capability(8, 0))
PY

echo ""
echo "=== [4] isolated venv for TitanRL deps (do NOT clobber DLAMI torch) ==="
python3 -m venv "$HOME/titanrl-venv" 2>&1 | tail -2
# shellcheck disable=SC1091
source "$HOME/titanrl-venv/bin/activate"
python -m pip install -q -U pip 2>&1 | tail -2
echo "venv python: $(python --version)"

echo ""
echo "=== [5] install probes (each is a CUDA dependency named in rl/README.md) ==="
probe_install() {
  local pkg="$1" t="$2" out rc
  echo ""
  echo "########## pip install $pkg ##########"
  out=$(timeout "$t" python -m pip install "$pkg" 2>&1); rc=$?
  echo "$out" | tail -30
  echo "########## pip install $pkg -> exit $rc ##########"
}
probe_install "torchstore" 600
probe_install "torchmonarch" 600
probe_install "vllm" 900
probe_install "flash-attn" 300
probe_install "attn-gym[linear,cudnn]" 300

echo ""
echo "=== [6] import / init probes ==="
python - <<'PY'
import sys, traceback

def attempt(name, fn):
    print(f"\n--- {name} ---")
    try:
        r = fn()
        print("OK:", r)
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e)[:2000])
        traceback.print_exc()

attempt("torch version + cuda.is_available",
        lambda: (__import__("torch").__version__, __import__("torch").cuda.is_available()))

attempt("import vllm", lambda: __import__("vllm").__version__)

attempt("import torchstore", lambda: __import__("torchstore").__version__)

attempt("import torchmonarch", lambda: __import__("torchmonarch").__version__)

def vllm_platform_probe():
    import vllm
    from vllm.platforms import current_platform
    return f"current_platform = {current_platform}"

attempt("vllm current_platform", vllm_platform_probe)

def vllm_device_probe():
    import vllm
    from vllm.config import DeviceConfig
    dc = DeviceConfig("auto")
    return f"DeviceConfig('auto').device = {dc.device}"

attempt("vllm DeviceConfig device", vllm_device_probe)

def vllm_engineargs_probe():
    from vllm.engine.arg_utils import EngineArgs
    ea = EngineArgs(model="facebook/opt-125m", enforce_eager=True)
    return f"EngineArgs ok: model={ea.model}, device={ea.device}"

attempt("vllm EngineArgs (no build)", vllm_engineargs_probe)
PY

echo ""
echo "=== [7] DONE $(date -u) ==="
echo "TITANRL_PROBE_COMPLETE"

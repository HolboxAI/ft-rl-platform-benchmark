#!/usr/bin/env bash
# Venv that runs DS-1000's tests (Linux x86_64): python 3.10 and DS-1000's pinned libraries, with CPU-only torch
# 2.2.0 and tensorflow-cpu 2.16.1. It holds no torch_tpu / libtpu, so a test program cannot open a TPU chip.
#
#   bash skyrl_ds1000/setup_exec_venv.sh            # -> ~/.venv-ds1000-exec
#   export DS1000_PYTHON=~/.venv-ds1000-exec/bin/python
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=${DS1000_EXEC_VENV:-$HOME/.venv-ds1000-exec}
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
uv venv --clear --python 3.10 "$VENV"
uv pip install --python "$VENV/bin/python" torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements-exec.txt"
"$VENV/bin/python" - <<'EOF'
import importlib
for m in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch", "tensorflow", "seaborn", "statsmodels", "xgboost", "gensim"]:
    print(f"{m:12s} {importlib.import_module(m).__version__}")
try:
    import torch_tpu  # noqa: F401
    raise SystemExit("torch_tpu is importable in the exec venv; it must not be")
except ImportError:
    pass
EOF
# Sample data that 15 problems load; the tests run without a network, so it is fetched here once, read-only.
DATA_HOME=${DS1000_DATA_HOME:-$HOME/.cache/ds1000-data}
[ -d "$DATA_HOME" ] && chmod -R u+w "$DATA_HOME"
SEABORN_DATA="$DATA_HOME/seaborn" SCIKIT_LEARN_DATA="$DATA_HOME/sklearn" "$VENV/bin/python" - <<'EOF'
import seaborn
from sklearn.datasets import fetch_california_housing
for name in ["penguins", "tips", "exercise", "planets"]:
    print(f"seaborn {name}: {seaborn.load_dataset(name).shape}")
print(f"sklearn california_housing: {fetch_california_housing().data.shape}")
EOF
chmod -R a-w "$DATA_HOME"
echo "DS1000_PYTHON=$VENV/bin/python"
echo "DS1000_DATA_HOME=$DATA_HOME"

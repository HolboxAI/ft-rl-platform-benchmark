#!/usr/bin/env bash
# Separate CPU-only Python env for running DS-1000 tests, pinned to the versions in
# data/ds1000/environment.yml (minus Matplotlib/TensorFlow, which the subset excludes).
# Kept apart from the training env so Neuron/CUDA package pins can't change test results.
set -euo pipefail
ENV="${1:-$HOME/ds1000-env}"
python3 -m venv "$ENV"
"$ENV/bin/pip" install -q --upgrade pip
"$ENV/bin/pip" install -q numpy==1.26.4 pandas==1.5.3 scipy==1.12.0 scikit-learn==1.4.0
"$ENV/bin/pip" install -q torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
"$ENV/bin/python" -c "import numpy, pandas, scipy, sklearn, torch; print('ds1000 env ok')"
echo "export REWARD_PYTHON=$ENV/bin/python"

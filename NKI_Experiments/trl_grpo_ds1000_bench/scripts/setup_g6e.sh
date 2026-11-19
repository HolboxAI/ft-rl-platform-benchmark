#!/usr/bin/env bash
# One-time setup on a g6e instance (AWS Deep Learning AMI, Ubuntu, with PyTorch + CUDA).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install -q -r requirements.txt
bash scripts/setup_reward_env.sh
python3 -c "import torch, trl; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'trl', trl.__version__)"

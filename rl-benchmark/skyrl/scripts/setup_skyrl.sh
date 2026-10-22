#!/bin/bash
# Clone SkyRL at the audited commit and prepare the client environment.
# Run on the accelerator host from the repo root:
#
#   bash skyrl/scripts/setup_skyrl.sh            # -> ~/SkyRL at $SKYRL_REF
#
# The server side is installed lazily by `uv run --extra ...` in each platform's
# start_server.sh. The client (tinker + tinker_cookbook) goes in its own venv:
# SkyRL pins tinker<0.28 for the server, and the Tinker leg used tinker 0.33.1 /
# tinker-cookbook 0.5.7 — record which client versions actually talk to the server.
set -euo pipefail
cd "$(dirname "$0")/../.."

SKYRL_DIR=${SKYRL_DIR:-$HOME/SkyRL}
SKYRL_REF=${SKYRL_REF:-b304652}   # commit audited in skyrl/metrics/porting_audit.json (2026-10-21)

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d "$SKYRL_DIR/.git" ]; then
  git clone https://github.com/NovaSky-AI/SkyRL "$SKYRL_DIR"
fi
git -C "$SKYRL_DIR" fetch --depth 1 origin "$SKYRL_REF" 2>/dev/null || true
git -C "$SKYRL_DIR" checkout "$SKYRL_REF"

# tinker-cookbook needs Python >= 3.11; TPU / Neuron images often ship 3.10, so let uv fetch 3.12.
uv venv --python 3.12 .venv-skyrl-client
uv pip install --python .venv-skyrl-client/bin/python \
  "${TINKER_PIN:-tinker>=0.25,<0.28}" "${COOKBOOK_PIN:-tinker-cookbook==0.5.7}" pandas pyarrow pillow
mkdir -p skyrl/logs
uv pip list --python .venv-skyrl-client/bin/python 2>/dev/null | grep -iE "^tinker|cookbook|torch|transformers" \
  | tee skyrl/logs/client_environment.txt
echo "SkyRL @ $(git -C "$SKYRL_DIR" log -1 --format='%h %cs')" | tee -a skyrl/logs/client_environment.txt

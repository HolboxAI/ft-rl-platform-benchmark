#!/usr/bin/env bash
# On a clean Linux VM (Ubuntu 22.04, x86_64): build the DS-1000 exec venv, clone DS-1000, run check_reward.py.
#   bash ~/skyrl_ds1000/run_check_vm.sh        # logs + results in ~/ds1000/check
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W=$HOME/ds1000; mkdir -p $W/check; cd $W
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/check/progress.log; }

log "== setup"
sudo apt-get update -qq && sudo apt-get install -y -qq git curl build-essential > $W/check/apt.log 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH" UV_HTTP_TIMEOUT=300
bash "$HERE/setup_exec_venv.sh" > $W/check/exec_venv.log 2>&1 || { log "FAILED exec venv"; exit 1; }
log "exec venv: $(grep -E '^(numpy|pandas|torch|tensorflow) ' $W/check/exec_venv.log | tr '\n' ' ')"
export DS1000_PYTHON=$HOME/.venv-ds1000-exec/bin/python
uv venv -q --clear --python 3.10 $W/.venv-driver && uv pip install -q --python $W/.venv-driver/bin/python pandas pyarrow
rm -rf DS-1000 && git clone -q --depth 1 https://github.com/xlang-ai/DS-1000.git && log "DS-1000 @ $(git -C DS-1000 rev-parse --short HEAD)"

log "== check_reward"
$W/.venv-driver/bin/python "$HERE/check_reward.py" --ds1000_repo $W/DS-1000 --out_dir $W/check > $W/check/check_reward.log 2>&1
log "check_reward exit $?"
grep -E '^\[(PASS|FAIL)\]|ALL PASS|FAILED' $W/check/check_reward.log | tee -a $W/check/progress.log
log "== done"

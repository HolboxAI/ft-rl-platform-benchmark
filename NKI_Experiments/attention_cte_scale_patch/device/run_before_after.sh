#!/usr/bin/env bash
# Run the patch's new attention_cte tests on a Trainium instance against BOTH kernel versions.
# Worked on a trn2.48xlarge (Neuron DLAMI Ubuntu 24.04, venv aws_neuronx_venv_torch_neuronx_2_14_0).
#
#   bash run_before_after.sh <path to nki-library-main on the instance>
#
# Trees: before = original kernel + the PATCHED test file (so both run the same cases),
#        after  = patched kernel + patched tests.
# Things that were needed on the instance (each cost time the first run):
#   * put the venv's bin on PATH, otherwise the NKI compile step runs `python None` and every test fails;
#   * load the repo's pytest plugin with `-p test.utils.pytest_plugin` (its options are not auto-registered
#     without `pip install -e .`);
#   * install allpairspy pytest-timeout pytest-xdist pytest-rerunfailures pytest-html pytest-metadata plotext
#     aws-embedded-metrics coverage pytest-cov httpx[http2] fabric2 paramiko websocket-client debugpy
#     pytest-memray "griffe>=2,<3" into a --target dir and add it to PYTHONPATH.
set -euo pipefail

REPO="${1:?path to nki-library-main}"
V=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
EXTRA="$HOME/extra"
WORK="$HOME/work"; RES="$HOME/results"
mkdir -p "$WORK" "$RES"

[ -d "$EXTRA" ] || "$V/bin/pip" install -q --target "$EXTRA" allpairspy pytest-timeout pytest-xdist \
  pytest-rerunfailures pytest-html pytest-metadata plotext aws-embedded-metrics coverage pytest-cov \
  "httpx[http2]" fabric2 paramiko websocket-client debugpy pytest-memray "griffe>=2,<3"

rm -rf "$WORK/after" "$WORK/before_newtests"
cp -r "$REPO" "$WORK/before_newtests"; cp -r "$REPO" "$WORK/after"
(cd "$WORK/after" && git init -q 2>/dev/null || true; git apply attention_cte_fp32_scale_in_exp.patch)
T=test/integration/nkilib/core/attention/test_attention_cte.py
cp "$WORK/after/$T" "$WORK/before_newtests/$T"          # same test cases for both kernels

run() {  # run <tree> <label> <cores>
  ( export PATH="$V/bin:$PATH" VIRTUAL_ENV="$V" NEURON_RT_VISIBLE_CORES="$3"
    cd "$WORK/$1"; export PYTHONPATH="$EXTRA:$WORK/$1/src:$WORK/$1"
    python -m pytest -c test/pytest.ini -p test.utils.pytest_plugin "$T" \
      -k "0.08838834764831845 or 0.10206207261596575" \
      --test-mode compile-and-infer --platform-target trn2 \
      --output-directory "$RES/out_$2" --junitxml "$RES/$2.xml" -q -p no:cacheprovider \
      > "$RES/$2.log" 2>&1 || true ) &
}
run before_newtests before 4-7     # separate NeuronCores so the two runs do not collide
run after after 0-3
wait
echo "junit + logs in $RES (before.xml, after.xml). Per-case QoR/cycles are under $RES/out_*/."

#!/usr/bin/env bash
# go.sh <overlay tag> <logname> <script> [args...]
V=/opt/aws_neuronx_venv_torch_neuronx_2_14_0; t=$1; L=$2; shift 2
export PATH="$V/bin:$PATH" VIRTUAL_ENV=$V NEURON_RT_VISIBLE_CORES=0-3 TMPDIR=~/e2e/tmp_$L PYTHONPATH=~/overlay_$t:~/hf
mkdir -p $TMPDIR; cd ~/e2e; timeout 900 python "$@" > ~/e2e/$L.log 2>&1; echo "$L exit $?"

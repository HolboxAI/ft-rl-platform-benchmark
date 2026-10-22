#!/bin/bash
# From the laptop: create a TPU VM, copy this repo, run the SkyRL probes, fetch results, delete the VM.
#
#   PROJECT=my-project ZONE=us-east5-b bash skyrl/tpu/launch_from_laptop.sh            # create + run
#   PROJECT=... ZONE=... bash skyrl/tpu/launch_from_laptop.sh fetch                    # pull results
#   PROJECT=... ZONE=... bash skyrl/tpu/launch_from_laptop.sh delete                   # stop billing
#
# Phase 1 probes Qwen/Qwen3-0.6B (from Hugging Face, no AWS access needed).
# The CholecT50 checkpoint probe needs the 9.9 GB SFT checkpoint on the VM; see
# skyrl/README.md ("TPU: getting the SFT checkpoint") before adding it to MODELS.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT:?set PROJECT}" "${ZONE:?set ZONE}"
ACCEL=${ACCEL:-v6e-1}
NAME=${NAME:-skyrl-tpu-probe}
MODELS=${MODELS:-Qwen/Qwen3-0.6B}
G=(gcloud --project "$PROJECT")
SSH=("${G[@]}" compute tpus tpu-vm ssh "$NAME" --zone "$ZONE")

case "${1:-run}" in
  fetch)
    mkdir -p skyrl/logs/tpu_fetch
    "${G[@]}" compute tpus tpu-vm scp --zone "$ZONE" --recurse "$NAME:~/rl-benchmark/skyrl/logs/*" skyrl/logs/tpu_fetch/
    ls skyrl/logs/tpu_fetch; exit 0 ;;
  delete)
    "${G[@]}" compute tpus tpu-vm delete "$NAME" --zone "$ZONE" --quiet; exit 0 ;;
esac

PROJECT=$PROJECT ZONE=$ZONE ACCEL=$ACCEL NAME=$NAME bash torchtpu/scripts/launch_tpu_vm.sh
tarball=$(mktemp -d)/repo.tgz
tar czf "$tarball" --exclude ".venv*" --exclude "*/logs/*" --exclude "data/cholect50/parquet" --exclude ".DS_Store" .
"${G[@]}" compute tpus tpu-vm scp --zone "$ZONE" "$tarball" "$NAME:~/repo.tgz"
"${SSH[@]}" --command "mkdir -p ~/rl-benchmark && tar xzf ~/repo.tgz -C ~/rl-benchmark && \
  cd ~/rl-benchmark && MODELS='$MODELS' nohup bash skyrl/tpu/run_all_probes.sh > ~/probes.out 2>&1 &"
echo "started on $NAME. Follow:  ${SSH[*]} --command 'tail -f ~/probes.out'"
echo "then:  PROJECT=$PROJECT ZONE=$ZONE bash skyrl/tpu/launch_from_laptop.sh fetch   (and 'delete' when done)"

#!/usr/bin/env bash
# TPU VM lifecycle for the SkyRL CholecT50 replication. Runs on the laptop.
#
#   bash gcloud/vm.sh create            # both VMs (spot by default; see config/tpu.env)
#   bash gcloud/vm.sh create rollout    # one of them: trainer | rollout
#   bash gcloud/vm.sh status
#   bash gcloud/vm.sh sync              # copy tpu_rl_benchmark/ (skyrl-tpu/ + miles-tpu/) + the SkyRL-tpu worktree to both VMs
#   bash gcloud/vm.sh ssh trainer [-- command]
#   bash gcloud/vm.sh fetch             # pull logs/ and metrics/ back
#   bash gcloud/vm.sh delete            # stop billing (both)
#   bash gcloud/vm.sh single-start      # single-VM layout: sync + start ops/single_vm.sh on the trainer
#   MILES=1 bash gcloud/vm.sh single-start   # the same for the Miles run (miles-tpu/ops/single_vm.sh)
#                                       # (setup, rollout server on chip 6, probes, run); laptop can go off after
#
# Spot VMs can be preempted at any time; `status` shows PREEMPTED, then `delete` +
# `create` again. TPU VMs bill while they exist, even when idle.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
# shellcheck disable=SC1091
source "$ROOT/config/tpu.env"
REPO_ROOT="$(dirname "$ROOT")"   # tpu_rl_benchmark/ (ROOT is its skyrl-tpu/)
SKYRL_TPU_LOCAL=${SKYRL_TPU_LOCAL:-$(dirname "$REPO_ROOT")/SkyRL-tpu}
G=("$GCLOUD" --project "$PROJECT")

name_of() { case "$1" in trainer) echo "$TRAINER_VM" ;; rollout) echo "$ROLLOUT_VM" ;; *) echo "unknown role $1" >&2; exit 1 ;; esac; }
accel_of() { case "$1" in trainer) echo "$TRAINER_ACCEL" ;; rollout) echo "$ROLLOUT_ACCEL" ;; esac; }
roles() { if [ -n "${1:-}" ]; then echo "$1"; else echo "trainer rollout"; fi; }

cmd=${1:-status}; shift || true
case "$cmd" in
  create)
    for r in $(roles "${1:-}"); do
      n=$(name_of "$r"); a=$(accel_of "$r")
      echo "creating $n ($a, zone $ZONE, spot=$SPOT)"
      "${G[@]}" compute tpus tpu-vm create "$n" --zone "$ZONE" \
        --accelerator-type "$a" --version "$RUNTIME_VERSION" \
        $( [ "$SPOT" = 1 ] && echo --spot ) \
        --scopes https://www.googleapis.com/auth/cloud-platform \
        --labels "experiment=skyrl-cholect50,role=$r"
    done ;;
  status)
    "${G[@]}" compute tpus tpu-vm list --zone "$ZONE" \
      --format="table(name.basename(),acceleratorType.basename(),state,health,networkEndpoints[0].ipAddress:label=INTERNAL_IP,schedulingConfig.spot)" ;;
  ip)
    "${G[@]}" compute tpus tpu-vm describe "$(name_of "${1:?role}")" --zone "$ZONE" --format='value(networkEndpoints[0].ipAddress)' ;;
  sync)
    tmp=$(mktemp -d)
    tar czf "$tmp/tpu_rl_benchmark.tgz" -C "$(dirname "$REPO_ROOT")" --exclude 'logs/*' --exclude '__pycache__' --exclude '.venv*' --exclude 'miles-src' "$(basename "$REPO_ROOT")"
    # The SkyRL worktree is copied with its uncommitted patch applied (git-tracked files + the new ones).
    # No .git goes over, so stamp the base commit and ship the patch next to the code.
    base=$(git -C "$SKYRL_TPU_LOCAL" rev-parse --short HEAD)
    git -C "$SKYRL_TPU_LOCAL" add -N . && git -C "$SKYRL_TPU_LOCAL" diff > "$SKYRL_TPU_LOCAL/SKYRL_TPU_PATCH.diff"
    echo "$base+skyrl-tpu ($(shasum -a 256 "$SKYRL_TPU_LOCAL/SKYRL_TPU_PATCH.diff" | cut -c1-12))" > "$SKYRL_TPU_LOCAL/SKYRL_TPU_VERSION"
    (cd "$SKYRL_TPU_LOCAL" && git ls-files -co --exclude-standard | tar czf "$tmp/SkyRL-tpu.tgz" -T -)
    rm -f "$SKYRL_TPU_LOCAL/SKYRL_TPU_PATCH.diff" "$SKYRL_TPU_LOCAL/SKYRL_TPU_VERSION"; git -C "$SKYRL_TPU_LOCAL" reset -q
    for r in $(roles "${1:-}"); do
      n=$(name_of "$r")
      "${G[@]}" compute tpus tpu-vm scp --zone "$ZONE" "$tmp/tpu_rl_benchmark.tgz" "$tmp/SkyRL-tpu.tgz" "$n:~/"
      "${G[@]}" compute tpus tpu-vm ssh "$n" --zone "$ZONE" --command \
        "set -e; rm -rf ~/tpu_rl_benchmark.new && mkdir -p ~/tpu_rl_benchmark.new ~/SkyRL-tpu && tar xzf ~/tpu_rl_benchmark.tgz -C ~/tpu_rl_benchmark.new && rsync -a ~/tpu_rl_benchmark.new/tpu_rl_benchmark/ ~/tpu_rl_benchmark/ && rm -rf ~/tpu_rl_benchmark.new && tar xzf ~/SkyRL-tpu.tgz -C ~/SkyRL-tpu && echo synced to \$(hostname)"
    done
    rm -rf "$tmp" ;;
  ssh)
    r=${1:?role}; shift || true
    [ "${1:-}" = "--" ] && shift
    if [ $# -gt 0 ]; then "${G[@]}" compute tpus tpu-vm ssh "$(name_of "$r")" --zone "$ZONE" --command "$*"
    else "${G[@]}" compute tpus tpu-vm ssh "$(name_of "$r")" --zone "$ZONE"; fi ;;
  fetch)
    for r in $(roles "${1:-}"); do
      n=$(name_of "$r"); dst="$ROOT/logs/fetched/$r"; mkdir -p "$dst"
      "${G[@]}" compute tpus tpu-vm scp --zone "$ZONE" --recurse "$n:~/tpu_rl_benchmark/skyrl-tpu/logs" "$n:~/tpu_rl_benchmark/skyrl-tpu/metrics" "$dst/" || true
    done
    echo "fetched into $ROOT/logs/fetched/" ;;
  single-start)
    bash "$0" sync trainer
    if [ "${MILES:-0}" = 1 ]; then  # the Miles run (miles-tpu/): same VM, same data
      "${G[@]}" compute tpus tpu-vm ssh "$TRAINER_VM" --zone "$ZONE" --command \
        "cd ~/tpu_rl_benchmark/miles-tpu && mkdir -p logs && SKIP_PROBES=${SKIP_PROBES:-0} RUN_ARGS='${RUN_ARGS:-}' nohup bash ops/single_vm.sh > logs/single_vm.out 2>&1 < /dev/null & disown; echo miles single_vm.sh started on \$(hostname)"
      echo "progress: gcloud storage cat gs://tpu-rl-run-cholec/runs/single_vm_miles/single_vm.log"; exit 0
    fi
    "${G[@]}" compute tpus tpu-vm ssh "$TRAINER_VM" --zone "$ZONE" --command \
      "cd ~/tpu_rl_benchmark/skyrl-tpu && mkdir -p logs && ENGINE=${ENGINE:-vllm} SKIP_PROBES=${SKIP_PROBES:-0} RUN_ARGS='${RUN_ARGS:-}' nohup bash ops/single_vm.sh > logs/single_vm.out 2>&1 < /dev/null & disown; echo single_vm.sh started on \$(hostname)"
    echo "progress: bash gcloud/run_status.sh   (works with the laptop off afterwards)" ;;
  delete)
    for r in $(roles "${1:-}"); do
      "${G[@]}" compute tpus tpu-vm delete "$(name_of "$r")" --zone "$ZONE" --quiet || true
    done ;;
  *) sed -n 2,14p "$0"; exit 1 ;;
esac

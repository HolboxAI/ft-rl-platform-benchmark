# Environment for the SkyRL trainer on the TPU trainer VM. Source it; do not run it.
#
#   source skyrl_cholect50/tpu_env.sh
#
# Override any of these before sourcing.
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SKYRL_TPU_DIR:=$HOME/SkyRL-tpu}"            # SkyRL @2c54cefc + skyrl-tpu patch
: "${SKYRL_VENV:=$HOME/.venv-skyrl-tpu}"         # built by setup/setup_trainer_vm.sh
# TOPOLOGY=two-vm (default): trainer v6e-8 (policy 0-3, ref 4-7) + rollout on its own v6e-1.
# TOPOLOGY=single: everything on the v6e-8 (policy 0-3, ref 4, rollout server on chip 6,
#                  started by ops/single_vm.sh); weights handed over through /dev/shm.
: "${TOPOLOGY:=two-vm}"
if [ "$TOPOLOGY" = single ]; then
  : "${POLICY_CHIPS:=0,1,2,3}"
  # 1 chip: 2-chip TorchTPU groups fail START_SESSION on v6e-8 (P0, 2026-10-24); 1 and 4 chips work.
  : "${REF_CHIPS:=4}"
  # The 1-chip ref ran out of contiguous HBM in step 1 (fp32 weights + 128 x ~2.3k-token forward):
  # keep its weights in host RAM and stream them per layer (FSDP CPU offload; same math).
  export REF_CPU_OFFLOAD=${REF_CPU_OFFLOAD:-true}
  : "${ROLLOUT_URLS:=http://localhost:8000}"
  export SKYRL_RELOAD_TRANSPORT=${SKYRL_RELOAD_TRANSPORT:-path}
  export SKYRL_RELOAD_EXPORT_DIR=${SKYRL_RELOAD_EXPORT_DIR:-/dev/shm/skyrl_reload}
else
  : "${POLICY_CHIPS:=0,1,2,3}"                   # v6e-8: policy on 4 chips ...
  : "${REF_CHIPS:=4,5,6,7}"                      # ... ref (KL 0.04) on the other 4
  : "${ROLLOUT_URLS:=http://rollout-vm:8000}"    # comma-separated rollout servers
fi
: "${RELOAD_AGENT_URLS:=}"                       # default: each rollout URL with port + 1
export TOPOLOGY ROLLOUT_URLS
export NUM_POLICY_CHIPS=${NUM_POLICY_CHIPS:-$(awk -F, '{print NF}' <<< "$POLICY_CHIPS")}
export NUM_REF_CHIPS=${NUM_REF_CHIPS:-$(awk -F, '{print NF}' <<< "$REF_CHIPS")}

export SKYRL_DEVICE=tpu
# No TorchTPU autoload: otherwise every process that imports torch (SkyRL's DataLoader workers,
# the driver) can open a chip and collide with the training workers (first run, 2026-10-24).
# The policy/ref workers register the device explicitly (skyrl-tpu patch, device.py).
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# transformers' hub kernels for the Gated DeltaNet layers are CUDA builds; use the
# pure-PyTorch path (the GPU run also set this, to avoid broken sm86 kernels).
export USE_HUB_KERNELS=NO
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$_HERE${PYTHONPATH:+:$PYTHONPATH}"
export SKYRL_RELOAD_AGENT_URLS="$RELOAD_AGENT_URLS"
export SKYRL_RELOAD_EXPORT_DIR="${SKYRL_RELOAD_EXPORT_DIR:-/tmp/skyrl_reload}"
export SKYRL_RELOAD_TRANSPORT="${SKYRL_RELOAD_TRANSPORT:-http}"

# Per-group TorchTPU launch env (see tpu_groups.py; UNVERIFIED until probes/p0 passes).
eval "$("$SKYRL_VENV/bin/python" "$_HERE/tpu_groups.py" --group "policy:$POLICY_CHIPS" --group "ref:$REF_CHIPS")"

export PATH="$SKYRL_VENV/bin:$PATH"
unset _HERE

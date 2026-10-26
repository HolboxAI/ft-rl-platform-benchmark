# Google Cloud settings for the verl CholecT50 LoRA RL replication (laptop).
GCLOUD=${GCLOUD:-/opt/homebrew/bin/gcloud}
PROJECT=${PROJECT:-tpu-rl-run}
ZONE=${ZONE:-us-east5-b}
SPOT=${SPOT:-1}                          # 1 = spot (preemptible)

# Single-VM topology (as in the SkyRL TPU run): one v6e-8.
# policy 0-3, ref 4, rollout 6. No separate rollout VM.
TRAINER_VM=${TRAINER_VM:-verl-cholec-trainer}
TRAINER_ACCEL=${TRAINER_ACCEL:-v6e-8}
ROLLOUT_VM=${ROLLOUT_VM:-verl-cholec-rollout}
ROLLOUT_ACCEL=${ROLLOUT_ACCEL:-none}
RUNTIME_VERSION=${RUNTIME_VERSION:-v2-alpha-tpuv6e}

REMOTE_ROOT=${REMOTE_ROOT:-/home/\$USER}
MODEL_REMOTE=${MODEL_REMOTE:-models/cholec-sft-qwen3.5-4b}

# Data already staged in GCS.
DATA_GCS=${DATA_GCS:-gs://tpu-rl-run-cholec/cholec}

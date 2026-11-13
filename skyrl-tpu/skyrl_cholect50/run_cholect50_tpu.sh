#!/usr/bin/env bash
# CholecT50 GRPO (RLVR) on SkyRL, Cloud TPU. Runs on the trainer VM.
#
# Replicates ft-rl-platform-benchmark-git/skyrl/scripts/run_cholec.sh (the GPU run:
# g5.12xlarge, SkyRL git 2c54cefc, reward 0.329 -> collapsed to 0.0). Every
# algorithm, data and optimiser override below is copied from that script. The
# lines marked [TPU] are the only differences; config/PARITY.md explains each.
#
#   bash skyrl_cholect50/run_cholect50_tpu.sh                          # full run, 1 epoch (16 steps)
#   bash skyrl_cholect50/run_cholect50_tpu.sh trainer.max_training_steps=1   # smoke test
#
# Needs: setup/setup_trainer_vm.sh done, data fetched, and the rollout VM's
# supervisor healthy at $ROLLOUT_URLS (rollout/start_rollout.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
# shellcheck disable=SC1091
source "$HERE/tpu_env.sh"

# Refuse to launch if this script drifted from the GPU run's documented config
# (ft-rl-platform-benchmark-git/skyrl/config/SKYRL_CONFIG.md); see config/PARITY.md.
PARITY_ARGS=(--set "trainer.placement.policy_num_gpus_per_node=${NUM_POLICY_CHIPS:-4}"
             --set "trainer.placement.critic_num_gpus_per_node=${NUM_POLICY_CHIPS:-4}"
             --set "trainer.placement.ref_num_gpus_per_node=${NUM_REF_CHIPS:-4}")
python3 "$HERE/check_config_parity.py" "${PARITY_ARGS[@]}" > /dev/null \
  || { python3 "$HERE/check_config_parity.py" "${PARITY_ARGS[@]}" | grep -v '^same'; exit 3; }
[ $# -gt 0 ] && echo "NOTE: extra overrides beyond the GPU config: $*" >&2

: "${DATA_DIR:=$HOME/cholec/skyrl_parquet}"
: "${MODEL_PATH:=$HOME/models/cholec-sft-qwen3.5-4b}"
: "${NUM_POLICY_CHIPS:=4}"
: "${NUM_REF_CHIPS:=4}"
: "${EXPORT_PATH:=$HOME/exports/cholec_skyrl_tpu}"
: "${CKPT_PATH:=$HOME/ckpts/cholec_skyrl_tpu}"
: "${CPU_OFFLOAD:=true}"   # as the GPU run; false ran out of HBM at the step-1 optimizer step on v6e (2026-10-24)
: "${RUN_ID:=cholect50-skyrl-tpu-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
# Every tracker.log() call is also written here (entrypoint.py), for analysis/plot_run.py.
export SKYRL_METRICS_JSONL="$LOG_DIR/metrics.jsonl"

IFS=, read -r -a _urls <<< "$ROLLOUT_URLS"
URLS_JSON="[$(printf "'%s'," "${_urls[@]}" | sed 's/,$//')]"
if [ "${#_urls[@]}" -eq 1 ]; then
  # One server: it is also the data-plane proxy, so SkyRL starts no router.
  PROXY_ARGS=("generator.inference_engine.external_proxy_url=${_urls[0]}")
else
  # Several servers: SkyRL starts its vllm-router in front of them.
  PROXY_ARGS=()
fi

for u in "${_urls[@]}"; do
  curl -sf -m 10 "$u/health" > /dev/null || { echo "rollout server $u is not healthy; start rollout/start_rollout.sh first" >&2; exit 2; }
done

# Ray: one "GPU" slot per TPU chip, so SkyRL's num_gpus=1 workers and {"GPU": n}
# placement groups schedule unchanged. Started here so the workers inherit this env.
ray stop --force > /dev/null 2>&1 || true
ray start --head --num-gpus=$((NUM_POLICY_CHIPS + NUM_REF_CHIPS)) --disable-usage-stats > "$LOG_DIR/ray_start.log" 2>&1
export RAY_ADDRESS=auto

{
  echo "run_id=$RUN_ID"; date -u +%FT%TZ
  echo "skyrl=$(cat "$SKYRL_TPU_DIR/SKYRL_TPU_VERSION" 2>/dev/null || git -C "$SKYRL_TPU_DIR" rev-parse --short HEAD)"
  curl -s -m 2 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type; echo
  python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
  python -c "import torch_tpu; print('torch_tpu', getattr(torch_tpu, '__version__', '?'))" || true
  for u in "${_urls[@]}"; do curl -s -m 5 "$u/version"; echo; done
  env | grep -E '^(SKYRL_|TORCH_TPU_|TPU_|USE_HUB)' | sort
} > "$LOG_DIR/environment.txt" 2>&1

cd "$SKYRL_TPU_DIR"
set -x
python "$HERE/entrypoint.py" \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/val.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.04 \
  trainer.algorithm.use_entropy_loss=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=false `# [TPU] one process per chip; GPU run: true` \
  trainer.placement.colocate_policy_ref=false `# [TPU] same reason` \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_CHIPS \
  trainer.placement.critic_num_gpus_per_node=$NUM_POLICY_CHIPS \
  trainer.placement.ref_num_gpus_per_node=$NUM_REF_CHIPS \
  generator.inference_engine.num_engines=${#_urls[@]} `# [TPU] external servers; GPU run: 4 local vLLM` \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=16 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=16 \
  trainer.remove_microbatch_padding=false \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=256 \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=1.0 \
  generator.max_turns=1 \
  trainer.policy.optimizer_config.lr=1e-5 \
  generator.inference_engine.backend=vllm `# [TPU] client protocol only; the server is rollout/rollout_server.py` \
  generator.inference_engine.run_engines_locally=false `# [TPU] GPU run: true` \
  generator.inference_engine.external_server_urls="$URLS_JSON" `# [TPU]` \
  "${PROXY_ARGS[@]}" \
  generator.inference_engine.weight_sync_backend=reload `# [TPU] GPU run: nccl (CUDA IPC when colocated)` \
  generator.batched=false \
  generator.vision_language_generator=true \
  environment.env_class=cholec \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.engine_init_kwargs.max_model_len=8192 \
  generator.inference_engine.max_num_seqs=64 \
  trainer.policy.fsdp_config.cpu_offload=$CPU_OFFLOAD \
  trainer.flash_attn=false `# [TPU] sdpa; flash-attn is CUDA-only` \
  trainer.ref.fsdp_config.cpu_offload=${REF_CPU_OFFLOAD:-false} `# [TPU single layout] 1-chip ref: weights in host RAM` \
  trainer.bf16=true \
  trainer.logger=console \
  trainer.project_name=cholec \
  trainer.run_name=cholec_rl_qwen3_5_4b_skyrl \
  trainer.resume_mode=none \
  trainer.log_path="$LOG_DIR/skyrl" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  "$@" 2>&1 | tee "$LOG_DIR/train.log"

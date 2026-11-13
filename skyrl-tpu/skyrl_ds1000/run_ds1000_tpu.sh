#!/usr/bin/env bash
# GRPO on DS-1000 with SkyRL, Cloud TPU, one v6e-8 (single-VM layout). Runs on the TPU VM.
#
# Layout and TPU settings are the CholecT50 run's (skyrl_cholect50/run_cholect50_tpu.sh, config/PARITY.md):
# policy FSDP2 on chips 0-3 with CPU offload, reference on chip 4 with CPU offload, HF rollout server on chip 6,
# `reload` weight sync through /dev/shm. The task settings are new:
#   - model Qwen/Qwen3.5-4B (the architecture the TPU patch was tested on), thinking off;
#   - text only: prompts <= 2048 tokens (DS-1000's longest is ~1400), replies <= 512 tokens;
#   - algorithm as SkyRL's GSM8K GRPO example: lr 1e-6, KL loss 0.001, one policy update per step
#     (policy_mini_batch_size = train_batch_size). The CholecT50 recipe (lr 1e-5, 8 updates per step)
#     collapsed after step 1 on GPU and TPU;
#   - eval on the 206 val problems before training (the base pass rate) and every EVAL_INTERVAL steps.
#
#   bash skyrl_ds1000/run_ds1000_tpu.sh                                   # MAX_STEPS steps (default 20)
#   bash skyrl_ds1000/run_ds1000_tpu.sh trainer.max_training_steps=1      # smoke test
#
# Needs: setup/setup_trainer_vm.sh, skyrl_ds1000/setup_exec_venv.sh, the parquet from skyrl_ds1000/data.py,
# the model, and the rollout server healthy at $ROLLOUT_URLS (skyrl_ds1000/single_vm_ds1000.sh does all of it).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
export TOPOLOGY=${TOPOLOGY:-single}
# shellcheck disable=SC1091
source "$ROOT/skyrl_cholect50/tpu_env.sh"
: "${DATA_DIR:=$HOME/ds1000/skyrl_parquet}"
: "${MODEL_PATH:=$HOME/models/Qwen3.5-4B}"
: "${MAX_STEPS:=20}"
: "${EVAL_INTERVAL:=10}"
: "${EXPORT_PATH:=$HOME/exports/ds1000_skyrl_tpu}"
: "${CKPT_PATH:=$HOME/ckpts/ds1000_skyrl_tpu}"
: "${CPU_OFFLOAD:=true}"
: "${RUN_ID:=ds1000-skyrl-tpu-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
export SKYRL_METRICS_JSONL="$LOG_DIR/metrics.jsonl"

# The reward runs DS-1000's tests in their own venv (skyrl_ds1000/sandbox.py). Exported before `ray start`,
# so the Ray worker that steps the environment has them.
export DS1000_PYTHON=${DS1000_PYTHON:-$HOME/.venv-ds1000-exec/bin/python}
export DS1000_DATA_HOME=${DS1000_DATA_HOME:-$HOME/.cache/ds1000-data}
export PYTHONPATH="$HERE:$PYTHONPATH"

IFS=, read -r -a _urls <<< "$ROLLOUT_URLS"
URLS_JSON="[$(printf "'%s'," "${_urls[@]}" | sed 's/,$//')]"
if [ "${#_urls[@]}" -eq 1 ]; then
  PROXY_ARGS=("generator.inference_engine.external_proxy_url=${_urls[0]}")
else
  PROXY_ARGS=()
fi
for u in "${_urls[@]}"; do
  curl -sf -m 10 "$u/health" > /dev/null || { echo "rollout server $u is not healthy" >&2; exit 2; }
done
ray stop --force > /dev/null 2>&1 || true
ray start --head --num-gpus=$((NUM_POLICY_CHIPS + NUM_REF_CHIPS)) --disable-usage-stats > "$LOG_DIR/ray_start.log" 2>&1
export RAY_ADDRESS=auto
{
  echo "run_id=$RUN_ID"; date -u +%FT%TZ
  echo "skyrl=$(cat "$SKYRL_TPU_DIR/SKYRL_TPU_VERSION" 2>/dev/null || git -C "$SKYRL_TPU_DIR" rev-parse --short HEAD)"
  curl -s -m 2 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type; echo
  python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
  python -c "import torch_tpu; print('torch_tpu', getattr(torch_tpu, '__version__', '?'))" || true
  "$DS1000_PYTHON" -c "import pandas, numpy, tensorflow, torch; print('ds1000 exec: pandas', pandas.__version__, 'numpy', numpy.__version__, 'tf', tensorflow.__version__, 'torch', torch.__version__)" 2>/dev/null
  for u in "${_urls[@]}"; do curl -s -m 5 "$u/version"; echo; done
  env | grep -E '^(SKYRL_|TORCH_TPU_|TPU_|USE_HUB|DS1000_)' | sort
} > "$LOG_DIR/environment.txt" 2>&1
cd "$SKYRL_TPU_DIR"
set -x
python "$HERE/ds1000_entrypoint.py" \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/val.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.use_entropy_loss=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_CHIPS \
  trainer.placement.critic_num_gpus_per_node=$NUM_POLICY_CHIPS \
  trainer.placement.ref_num_gpus_per_node=$NUM_REF_CHIPS \
  generator.inference_engine.num_engines=${#_urls[@]} \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.max_training_steps=$MAX_STEPS \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.remove_microbatch_padding=false \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=512 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.max_turns=1 \
  generator.chat_template_kwargs.enable_thinking=false \
  trainer.policy.optimizer_config.lr=1e-6 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.external_server_urls="$URLS_JSON" \
  "${PROXY_ARGS[@]}" \
  generator.inference_engine.weight_sync_backend=reload \
  generator.batched=false \
  generator.vision_language_generator=false \
  environment.env_class=ds1000 \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.engine_init_kwargs.max_model_len=4096 \
  generator.inference_engine.max_num_seqs=64 \
  trainer.policy.fsdp_config.cpu_offload=$CPU_OFFLOAD \
  trainer.flash_attn=false \
  trainer.ref.fsdp_config.cpu_offload=${REF_CPU_OFFLOAD:-true} \
  trainer.bf16=true \
  trainer.logger=console \
  trainer.project_name=ds1000 \
  trainer.run_name=ds1000_grpo_qwen3_5_4b_skyrl_tpu \
  trainer.resume_mode=none \
  trainer.log_path="$LOG_DIR/skyrl" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  trainer.ckpt_path="$CKPT_PATH" \
  "$@" 2>&1 | tee "$LOG_DIR/train.log"

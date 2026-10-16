# verl — CholecT50 GRPO / RLVR configuration

This is the exact configuration for the **RL / GRPO stage** only. The SFT stage
was produced on Tinker (see `../tinker/`); this stage loads that merged,
thinking-disabled checkpoint and fine-tunes the full model against the triplet-F1
reward.

## Shared

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-4B` |
| SFT checkpoint | Tinker output, LoRA merged, thinking-disabled (S3: `s3://stanford-train-data/models/cholec-sft-qwen3.5-4b`) |
| Data | CholecT50 — 256 train / 32 val frames (parquet) |
| Task | list `instrument-verb-target` action triplets per frame |
| Reward | `0.4 * triplet_F1 + 0.6 * mean(instrument_F1, verb_F1, target_F1)` |
| Precision | bfloat16 (model, ref, optimizer) |
| Container | `verlai/verl:uv.cu130` + 3 Qwen3.5 patches, rebuilt as `verl-cholec:fix` |

## RL / GRPO stage (verl)

### Algorithm

| Parameter | Value |
|---|---|
| `algorithm.adv_estimator` | `grpo` |
| `algorithm.use_kl_in_reward` | `False` |
| `actor_rollout_ref.actor.use_kl_loss` | `True` |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.04` |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` |
| `actor_rollout_ref.actor.entropy_coeff` | `0` |

### Data

| Parameter | Value |
|---|---|
| `data.train_files` | `train.parquet` (256 → 235 after filter) |
| `data.val_files` | `val.parquet` (32 → 28 after filter) |
| `data.image_key` | `images` |
| `data.image_patch_size` | `16` |
| `data.train_batch_size` | `16` |
| `data.max_prompt_length` | `2048` |
| `data.max_response_length` | `256` |
| `data.filter_overlong_prompts` | `True` |
| `data.truncation` | `error` |
| `+data.apply_chat_template_kwargs.enable_thinking` | `False` (the `+` is required) |

### Model / Actor (full-model FSDP, no LoRA)

| Parameter | Value |
|---|---|
| `actor_rollout_ref.model.use_remove_padding` | `False` |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `True` |
| `actor_rollout_ref.actor.strategy` | `fsdp` |
| `actor_rollout_ref.actor.optim.lr` | `1e-5` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `16` |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `1` |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `False` |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | `4` |
| `actor_rollout_ref.actor.fsdp_config.model_dtype` | `bfloat16` |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | `True` |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `True` |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `True` |
| `actor_rollout_ref.actor.fsdp_config.offload_policy` | `False` |
| `actor_rollout_ref.actor.use_torch_compile` | `False` |

### Rollout (vLLM)

| Parameter | Value |
|---|---|
| `actor_rollout_ref.rollout.name` | `vllm` |
| `actor_rollout_ref.rollout.n` | `8` (group size) |
| `actor_rollout_ref.rollout.temperature` | `0.7` |
| `actor_rollout_ref.rollout.top_p` | `1.0` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `1` |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.6` |
| `actor_rollout_ref.rollout.prompt_length` | `2048` |
| `actor_rollout_ref.rollout.response_length` | `256` |
| `actor_rollout_ref.rollout.max_model_len` | `4096` |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `4096` |
| `actor_rollout_ref.rollout.max_num_seqs` | `128` |
| `actor_rollout_ref.rollout.ignore_eos` | `False` |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | `False` |
| `actor_rollout_ref.rollout.enable_prefix_caching` | `False` |
| `actor_rollout_ref.rollout.free_cache_engine` | `True` |

### Trainer

| Parameter | Value |
|---|---|
| `trainer.total_epochs` | `1` → **14 steps** |
| `trainer.n_gpus_per_node` | `4` |
| `trainer.nnodes` | `1` |
| `trainer.val_before_train` | `True` |
| `trainer.save_freq` / `test_freq` | `16` |
| `trainer.resume_mode` | `disable` (avoid stale-checkpoint auto-resume) |
| `trainer.logger` | `["console"]` |
| `trainer.project_name` | `verl_cholec_grpo` |
| `trainer.experiment_name` | `cholec_rl_qwen3_5_4b` |
| `trainer.balance_batch` | `True` |

### Hardware / runtime

| Parameter | Value |
|---|---|
| Instance | AWS SageMaker `ml.g5.12xlarge` — 4× A10G 24 GB |
| Launch | `uv run --frozen --offline --all-packages --extra vllm --extra fsdp python3 -m verl.trainer.main_ppo` |
| Offline | `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |
| Peak GPU mem | ~19.7 GB / 24 GB per A10G |
| Wall time | ~90–100 min (~388 s / step) |

## Eval

| Parameter | Value |
|---|---|
| Val split | 28 frames (after `filter_overlong_prompts`) |
| Decoding | greedy (`do_sample=False`), no thinking |
| Metric | blended reward (see formula above) |
| Start (val_before_train) | 0.000 |
| Final (global_step_14) | **0.715** |

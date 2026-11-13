# SkyRL — CholecT50 GRPO / RLVR configuration

This is the exact configuration for the **RL / GRPO stage** only. The SFT stage
was produced on Tinker (see `../tinker/`); this stage loads that merged
checkpoint and fine-tunes the full model against the triplet-F1 reward. It
mirrors the verl leg (`../verl/`) one-to-one where the two frameworks allow.

## Shared

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-4B` |
| SFT checkpoint | Tinker output, LoRA merged (`~/cholec/merged`) |
| Data | CholecT50 — 256 train / 32 val frames (parquet) |
| Task | list `instrument-verb-target` action triplets per frame |
| Reward | `0.4 * triplet_F1 + 0.6 * mean(instrument_F1, verb_F1, target_F1)` |
| Precision | bfloat16 |
| Framework | SkyRL FSDP2 + vLLM |

## RL / GRPO stage (SkyRL)

### Algorithm

| Parameter | Value |
|---|---|
| `trainer.algorithm.advantage_estimator` | `grpo` |
| `trainer.algorithm.use_kl_loss` | `true` |
| `trainer.algorithm.kl_loss_coef` | `0.04` |
| `trainer.algorithm.use_entropy_loss` | `false` |
| `trainer.algorithm.loss_reduction` | `token_mean_legacy` |

### Data

| Parameter | Value |
|---|---|
| `data.train_data` | `train.parquet` (256) |
| `data.val_data` | `val.parquet` (32) |
| `trainer.train_batch_size` | `16` |
| `trainer.eval_batch_size` | `32` |
| `trainer.max_prompt_length` | `4096` |
| `generator.sampling_params.max_generate_length` | `256` |
| `generator.max_turns` | `1` |

### Model / Actor (full-model FSDP2, no LoRA)

| Parameter | Value |
|---|---|
| `trainer.strategy` | `fsdp` |
| `trainer.placement.colocate_all` | `true` |
| `trainer.placement.policy_num_gpus_per_node` | `4` |
| `trainer.placement.critic_num_gpus_per_node` | `4` |
| `trainer.placement.ref_num_gpus_per_node` | `4` |
| `trainer.policy.optimizer_config.lr` | `1e-5` |
| `trainer.policy_mini_batch_size` | `2` |
| `trainer.micro_forward_batch_size_per_gpu` | `1` |
| `trainer.micro_train_batch_size_per_gpu` | `1` |
| `trainer.update_epochs_per_batch` | `1` |
| `trainer.remove_microbatch_padding` | `false` |
| `trainer.policy.fsdp_config.cpu_offload` | `true` |
| `trainer.bf16` | `true` |

### Rollout (vLLM)

| Parameter | Value |
|---|---|
| `generator.n_samples_per_prompt` | `8` (group size) |
| `generator.sampling_params.temperature` | `0.7` |
| `generator.sampling_params.top_p` | `1.0` |
| `generator.inference_engine.backend` | `vllm` |
| `generator.inference_engine.num_engines` | `4` |
| `generator.inference_engine.tensor_parallel_size` | `1` |
| `generator.inference_engine.run_engines_locally` | `true` |
| `generator.inference_engine.weight_sync_backend` | `nccl` |
| `generator.inference_engine.gpu_memory_utilization` | `0.6` |
| `generator.inference_engine.max_num_seqs` | `64` |
| `generator.inference_engine.engine_init_kwargs.max_model_len` | `8192` |
| `generator.batched` | `false` |
| `generator.vision_language_generator` | `true` |

### Trainer

| Parameter | Value |
|---|---|
| `trainer.epochs` | `1` → 15 steps logged (256 / 16) |
| `trainer.eval_before_train` | `true` |
| `trainer.eval_interval` | `16` |
| `trainer.resume_mode` | `none` |
| `trainer.logger` | `console` |
| `trainer.project_name` | `cholec` |
| `trainer.run_name` | `cholec_rl_qwen3_5_4b_skyrl` |
| `trainer.ckpt_interval` | `0` (disabled — avoids ~53 GB full checkpoint) |
| `trainer.hf_save_interval` | `0` (disabled — avoids the HF-weights save that OOM'd the node; no epoch-end model save) |
| `trainer.dump_eval_results` | `true` |

### Hardware / runtime

| Parameter | Value |
|---|---|
| Instance | AWS EC2 `g5.12xlarge` — 4× A10G 24 GB, 186 GB RAM |
| Launch | `uv run --isolated --extra fsdp python examples/train/cholec/entrypoint.py …` |
| Env | `RAY_memory_usage_threshold=0.99`, `USE_HUB_KERNELS=NO`, `SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800` |
| Result | collapsed to 0.0 reward; final `save_hf_model` OOM'd (186 GB RAM) |

## Eval

| Parameter | Value |
|---|---|
| Val split | 32 frames |
| Decoding | `eval_before_train` greedy |
| Metric | blended reward (see formula above) |
| Start (`eval_before_train`) | **0.329** |
| Final (`eval_interval=16`) | **not reached** — blocked by the HF-save OOM |

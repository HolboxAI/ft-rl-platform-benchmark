# SkyRL CholecT50 — settings and findings

## Run settings (all platforms: GPU control, TPU, Trainium)

The client has two presets (`--preset`); any flag overrides its preset value.

| Setting | preset `tinker` | preset `skyrl-grpo` | Tinker leg | Existing SkyRL GPU run |
|---|---|---|---|---|
| Interface | Tinker API → SkyRL server | Tinker API → SkyRL server | Tinker API → Thinking Machines | SkyRL Python entrypoint |
| Backend | FSDP (GPU) / JAX (TPU, Trainium) | same | managed | FSDP2 + vLLM |
| Start weights | merged SFT checkpoint | merged SFT checkpoint | Tinker LoRA SFT weights | merged SFT checkpoint |
| Trained weights (`--lora-rank`) | LoRA 16 | **full model (0)** | LoRA 16 | full model |
| Loss (`--loss`) | `importance_sampling` | **`ppo`** → SkyRL `regular`, ε 0.2 / 0.2 | `importance_sampling` | `regular` (GRPO), ε 0.2 / 0.2 |
| Advantages (`--norm-adv-by-std`) | r − group mean | **(r − mean) / (std + 1e-6)** | r − group mean | (r − mean) / (std + 1e-6) |
| Updates per step (`--num-substeps`) | 1 | **8** | 1 | 8 (mini-batch 2 of 16 prompts) |
| Constant-reward groups | dropped | **kept** | dropped | kept |
| Temperature | 1.0 | **0.7** | 1.0 | 0.7 |
| KL | 0 | 0 (SkyRL's Tinker backend forces `use_kl_loss=False`) | 0 | **0.04** `low_var_kl` |
| Optimizer | Adam (0.9, 0.95), from cookbook | same | same | SkyRL defaults |
| Loss reduction | backend default | backend default | managed | `token_mean_legacy` |
| LR | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| Prompts × samples, steps | 16 × 8, 16 | 16 × 8, 16 | 16 × 8, 16 | 16 × 8, 15 logged |
| Max response tokens | 256 | 256 | 256 | 256 (prompt cap 4096) |
| Data | 256 / 32 (verl parquet) | 256 / 32 | 256 / 300 | 256 / 32 |
| Eval | greedy val, before + after | same | greedy val n=300 | `eval_before_train` only |
| Reward | `common/rewards/cholect50.py` | same | same formula | same formula |

Remaining gaps between `skyrl-grpo` and the existing run are KL (cannot be
enabled through the Tinker API), the optimizer betas, and the loss reduction.
The `ppo` loss through the Tinker API is documented as not yet validated.

## Server settings

| | GPU control | TPU | Trainium |
|---|---|---|---|
| Command | `skyrl.tinker.api --backend fsdp` | `uv run --extra tpu --extra tinker -m skyrl.tinker.api --backend jax` | `uv run --no-sync -m skyrl.tinker.api --backend jax` |
| backend-config | `remove_microbatch_padding=false`, 4 GPUs, CPU offload | `max_lora_rank=16, gradient_checkpointing=true` (+ `fully_sharded_data_parallel_size` on multi-chip) | same as TPU |
| Instance | g5.12xlarge (4× A10G) | ⏳ v6e-1 / v6e-4 | ⏳ trn1.2xlarge → trn1.32xlarge |
| Client versions | ⏳ (`logs/client_environment.txt`) | ⏳ | ⏳ |

## Probe results (`<chip>/run_probe.sh`)

| Date | Platform | Model | connect | train_text | rl_loss | sample_text | image_input |
|---|---|---|---|---|---|---|---|
| ⏳ | TPU | Qwen3-0.6B | | | | | |
| ⏳ | TPU | cholec-sft-qwen3.5-4b | | | | | |
| ⏳ | Trainium | Qwen3-0.6B | | | | | |
| ⏳ | GPU control | cholec-sft-qwen3.5-4b | | | | | |

## Findings log

| Date | Finding |
|---|---|
| 2026-10-22 | SkyRL's JAX `Qwen3_5ForCausalLM` is text-only (`Qwen3_5TextModel`); images are rendered only through vLLM's render endpoint (FSDP/Megatron backends). CholecT50 cannot run on the JAX backend as-is. |
| 2026-10-22 | `skyrl/tx` (JAX models) has 0 NVIDIA-specific call sites (25 files, 5.0k lines). |
| 2026-10-22 | The server's `backend` default is `megatron`; `--backend jax` must be passed explicitly. The API server forwards its `uv run` flags (e.g. `--extra tpu`, `--no-sync`) to the engine process. |
| 2026-10-22 | SkyRL pins `tinker>=0.25,<0.28` server-side; the Tinker leg's client was `tinker` 0.33.1. Client/server version compatibility is unverified. |

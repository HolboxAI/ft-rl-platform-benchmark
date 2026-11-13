# Qwen3-ASR — GRPO RL platform comparison (CUDA vs Trainium2)

This directory documents the **Qwen3-ASR-0.6B** reinforcement-learning (GRPO)
platform comparison: the *same* GRPO run, on the *same* 32-utterance dataset, on
both NVIDIA CUDA and AWS Trainium2 (native torch-neuronx).

The goal is a straight-to-the-point answer to: *does standard TRL/ms-swift GRPO
RL run correctly on Trainium2, and how does it compare to CUDA?*

## Headline result

| | CUDA (g6.xlarge, L4) | Trainium2 (trn2.48xlarge) |
|---|---|---|
| `train_runtime` | **211.28 s** (3 m 31 s) | **16,040.88 s** (4 h 27 m 21 s) |
| First step | 27.93 s (one-time warm-up) | 2,017 s (33.6 min graph compile) |
| Steady-state step | flat ~5.9 s | erratic 67 s → 1,297 s |
| Final reward | **−0.11055** | **−0.11055** (identical) |

**Trainium2 is 75.9× slower on this workload, but numerically identical.** The gap is
*per-shape NEFF graph recompilation* (compilation overhead), not compute throughput.

## Method (identical on both platforms)

- **Base model:** `Qwen/Qwen3-ASR-0.6B` (787.47 M params; 5.05 M trainable via LoRA = 0.64 %)
- **Framework:** ms-swift `swift rlhf --rlhf_type grpo` (wraps TRL `GRPOTrainer`)
- **Reward:** `r = −0.5·(WER+CER) − 0.05·length_penalty − 0.05·repetition`
- **Dataset:** 32 utterances (first 32 rows of `librispeech_asr` clean/train.100), byte-identical on both
- **Frozen:** audio tower (`thinker.audio_tower`) + aligner; LoRA on the LLM decoder
- **GRPO:** `num_generations=8`, lr 5e-7, grad_accum 8, max_completion_length 256, temperature 1.0, beta 0.0, 1 epoch → 32 steps
- **Precision:** fp16 (CUDA) vs bf16 (Trainium2)

## Key observations

- **Numerical parity** — final reward −0.11055 on both; per-step rewards near-identical → torch-neuronx computes the GRPO loss/gradients correctly.
- **Compilation, not compute** — the 75.9× gap is per-shape NEFF recompilation on variable-length completions (36–56 tokens). CUDA compiles once; Neuron compiles one graph per unique shape.
- **Harder cousin on big models** — the same root cause exhausts on-core TDRAM on a 4.95 B VLM (`trainium2/TRAINIUM_ISSUES_LOG.md` issue #23): small model → slow, big model → crash.
- **Memory telemetry gap** — ms-swift's `memory(GiB)` reports 0 on Neuron (CUDA-only metric).

## Reports

Full LaTeX reports (with charts) live in
[`HolboxAI/ft-rl-benchmark-results`](https://github.com/HolboxAI/ft-rl-benchmark-results)
under `qwen3-asr/`:

- `platform-rl-cuda.pdf` — CUDA run report
- `platform-rl-trn2.pdf` — Trainium2 run report
- `platform-rl-cuda-vs-trn2.pdf` — the side-by-side comparison

## SkyRL + vllm-neuron (in progress)

A follow-up leg is replacing the in-process native engine with the **vllm-neuron
fork** (real vLLM serving on Trainium) as the rollout engine, driven by SkyRL.
See `skyrl/` in the main repo and the `vllm_neuron` inference backend PR
(`susmit-holbox/SkyRL` PR #1).

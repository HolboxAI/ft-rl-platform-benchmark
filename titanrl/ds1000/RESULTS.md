# TitanRL DS-1000 GRPO — result

**Date:** 2026-11-02
**Hardware:** AWS `g5.12xlarge` (4× NVIDIA A10G 24GB), us-east-1
**Stack:** TitanRL (TorchTitan RL) — `torch 2.16.0+cu132`, `vllm 0.1.dev`, Monarch orchestration
**Model:** `Qwen3-0.6B` (text-only), GRPO
**Dataset:** DS-1000 (code completion), 800 train / 200 val, prompts > 5000 chars filtered

## Result

25 training steps completed before the run hit TitanRL's "10 consecutive
untrainable batches" guard. The reward **improves** over those steps:

| | early (steps 1–14) | late (steps 15–25) |
|---|---|---|
| `rollout_reward/_mean` | ~0.42 (0.30–0.52) | ~0.58 (0.48–0.64) |

Final reward at step 25: **0.64** (DS-1000 completion runs cleanly). Loss
(`loss/mean`) stays small and bounded (≈ −0.010 … +0.035).

Full per-step trajectory is in `ds1000_train.log`; plots in
`reward_vs_step.png` and `loss_vs_step.png`.

## Setup notes

- `data.py` / `env.py` / `rubric.py` — the DS-1000 experiment (single-turn
  code-completion `MessageEnv`, pass/fail `RewardFn` that runs the completion).
- `recipe.py` — `rl_grpo_qwen3_0_6b_ds1000` (`Controller.Config`), placed at
  `torchtitan_recipes/rl/ds1000.py`.
- `run_ds1000.sh` — launch command.

## Issues fixed during the run

1. `IterableRLDataLoader` → `IterableRolloutDataLoader` (class renamed upstream).
2. DS-1000 has no `id` column → use the row index.
3. vLLM `gpu_memory_utilization` default 0.9 OOM'd the generator → set
   `gpu_memory_limit=0.5`.
4. Qwen3-1.7B trainer OOM'd on a 22GB A10G → switched to **Qwen3-0.6B**.
5. DS-1000 prompts > 2048 tokens → `seq_len=4096` + a 5000-char prompt filter.
6. Empty rollout (`rollout.turns == []`) → rubric returns −0.5.

## Known limitation

The run stops at ~25 steps with
`RuntimeError: 10 consecutive untrainable batches — check reward diversity and
training-sample filters`. The reward is diverse (~55% pass / 45% fail), so the
cause is a training-sample filter dropping groups (not reward degeneracy); worth
investigating the batcher's `num_tokens_to_pack(s) <= seq_len` path before a
full 100-step run.

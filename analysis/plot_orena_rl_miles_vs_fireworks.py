#!/usr/bin/env python3
"""Overlay Miles vs Fireworks ORena VLM RL (GRPO) curves.

Same base model (Qwen3-VL-8B-Instruct), same 2000 rows, same 8 samples/prompt,
same 512-token cap, same temperature 0.7, same kl 0.04, same reward. The only
differences are the trainer (self-hosted Miles vs Fireworks managed RL) and the
per-step reporting granularity: Fireworks reports 10 chunks of 200 rows, Miles
reports 31 rollout steps (64 prompts x 8 samples each). Both x-axes are rescaled
to "% of training run" so the shapes are comparable.

Writes analysis/plots/orena_rl_miles_vs_fireworks.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

FIREWORKS = os.path.join(ROOT, "fireworks", "metrics", "final_results.json")
MILES = os.path.join(ROOT, "miles", "metrics", "orena_rl_final_results.json")

FW_BLUE = "#2563eb"
MI_PURPLE = "#7c3aed"


def pct(xs):
    """Rescale an index list to 0..100 (% of run)."""
    n = len(xs)
    if n <= 1:
        return [0.0] * n
    return [100.0 * i / (n - 1) for i in range(n)]


def rolling(xs, window):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


fw = json.load(open(FIREWORKS))["experiments"]["orena_rl"]
mi = json.load(open(MILES))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("ORena VLM RL / GRPO — Miles vs Fireworks (Qwen3-VL-8B-Instruct, 2000 rows, 1 epoch)",
             fontsize=12.5, fontweight="bold")

# Panel 1: exact_match — the strict 0/1 metric Fireworks reports.
ax1.plot(pct(fw["exact_match_curve"]), fw["exact_match_curve"],
         marker="o", ms=4, lw=1.5, color=FW_BLUE,
         label=f"Fireworks → {fw['exact_match_curve'][-1]:.3f}")
ax1.plot(pct(mi["exact_match_curve"]), mi["exact_match_curve"],
         lw=0.4, alpha=0.3, color=MI_PURPLE)
ax1.plot(pct(mi["exact_match_curve"]), rolling(mi["exact_match_curve"], 5),
         lw=1.8, color=MI_PURPLE,
         label=f"Miles (roll-5) → {mi['exact_match_curve'][-1]:.3f}")
ax1.set_title("exact_match (reported metric)", fontsize=10, fontweight="bold")
ax1.set_xlabel("% of training run")
ax1.set_ylabel("exact_match")
ax1.set_ylim(0, 0.5)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8)

# Panel 2: dense correctness (the GRPO training signal).
ax2.plot(pct(fw["score_curve"]), fw["score_curve"],
         marker="o", ms=4, lw=1.5, color=FW_BLUE,
         label=f"Fireworks → {fw['score_curve'][-1]:.3f}")
ax2.plot(pct(mi["score_curve"]), mi["score_curve"],
         lw=0.4, alpha=0.3, color=MI_PURPLE)
ax2.plot(pct(mi["score_curve"]), rolling(mi["score_curve"], 5),
         lw=1.8, color=MI_PURPLE,
         label=f"Miles (roll-5) → {mi['score_curve'][-1]:.3f}")
ax2.set_title("correctness (reward_value, training signal)", fontsize=10, fontweight="bold")
ax2.set_xlabel("% of training run")
ax2.set_ylabel("correctness")
ax2.set_ylim(0, 0.6)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8)

fig.text(0.5, 0.005,
         "x-axes rescale different step counts to % of run (Fireworks 10 chunks of 200 rows; "
         "Miles 31 steps of 64 prompts x 8 samples). Miles per-step is noisier (64 vs 200 unique prompts).",
         ha="center", fontsize=7.5, color="#444")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
out = os.path.join(OUT, "orena_rl_miles_vs_fireworks.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("wrote", out)

#!/usr/bin/env python3
"""Join this repo's CholecT50 runs with the GPU / managed-platform reference results.

    python3 analysis/compare.py            # table to stdout
    python3 analysis/compare.py --plot     # + analysis/plots/cholect50_rl_curves.png (needs matplotlib)

Reference numbers are copied from `ft-rl-platform-benchmark-main 2` and `nemoRL/`
(sources noted per row), so this script has no cross-repo path dependency.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (platform, hardware, start, final, steps, wall_s, cost_usd, note, source)
# Metric: blended reward = 0.4 * triplet F1 + 0.6 * mean(instrument, verb, target F1), greedy val.
REFERENCE = [
    ("tinker", "managed, LoRA 16", 0.364, 0.419, 16, None, None, "val n=300", "tinker/metrics/final_results.json"),
    ("verl", "4x A10G, full model", 0.000, 0.715, 14, 5400, 8.5, "thinking-disabled SFT start, val n=28",
     "verl/metrics/final_results.json"),
    ("miles", "4x A100 (p4d spot), full model", 0.319, 0.651, 16, 4193, 11.6, "val n=32",
     "miles/metrics/cholect50_rl_final_results.json"),
    ("nemo-rl", "4x A10G (g5.24xlarge), full model", 0.000, 0.471, 14, 10934, 24.7, "val n=28",
     "nemoRL/reports/full-1/eval_*.json"),
    ("skyrl", "4x A10G, full model", 0.329, 0.000, 15, 13500, 21.3, "collapsed; train reward",
     "skyrl/metrics/final_results.json"),
]


def load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "metrics", "*final_results.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for name, exp in d.get("experiments", {}).items():
            if name == "cholect50_rl":
                runs.append((d.get("platform"), exp, os.path.relpath(path, ROOT)))
    return runs


def fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def table(runs):
    print("\n## CholecT50 GRPO: blended reward on val (greedy)\n")
    print("| platform | hardware | val start → final | rollout reward start → final | steps | wall s | first / steady step s | $ | note |")
    print("|---|---|---|---|---|---|---|---|---|")
    for plat, hw, s, f_, steps, wall, cost, note, _src in REFERENCE:
        print(f"| {plat} (ref) | {hw} | {fmt(s)} → {fmt(f_)} | — | {fmt(steps)} | {fmt(wall)} | — | {fmt(cost, 2)} | {note} |")
    for plat, exp, _path in runs:
        acc = exp.get("accelerator", {})
        hw = f"{acc.get('instance_type') or '?'} ({acc.get('backend')})" if acc else exp.get("framework", "?")
        steps_s = f"{fmt(acc.get('first_step_s'), 0)} / {fmt(acc.get('steady_step_s'), 1)}" if acc else "—"
        note = f"lora {exp['lora_rank']}" if exp.get("lora_rank") else "full model"
        if exp.get("preset"):
            note += f", preset {exp['preset']}"
        print(f"| **{plat}** | {hw} | {fmt(exp.get('start_val_reward'))} → {fmt(exp.get('final_val_reward'))} | "
              f"{fmt(exp.get('start_score'))} → {fmt(exp.get('final_score'))} | {exp.get('steps')} | "
              f"{fmt(exp.get('wall_time_s'), 0)} | {steps_s} | {fmt(exp.get('estimated_cost_usd'), 2)} | {note} |")


def plot(runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for plat, exp, _ in runs:
        if exp.get("score_curve"):
            ax.plot(range(1, len(exp["score_curve"]) + 1), exp["score_curve"], label=plat, linewidth=1.2)
    for plat, _hw, _s, f_, *_ in REFERENCE:
        ax.axhline(f_, linestyle="--", linewidth=0.8, color="gray")
        ax.annotate(f"{plat} final val {f_:.3f}", (1, f_), fontsize=8, color="gray", va="bottom")
    ax.set_xlabel("GRPO step (16 frames × 8 samples)")
    ax.set_ylabel("rollout blended reward")
    ax.set_title("CholecT50 GRPO — Qwen3.5-4B (SFT checkpoint)")
    if runs:
        ax.legend(loc="lower right", fontsize=8)
    out = os.path.join(ROOT, "analysis", "plots", "cholect50_rl_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    runs = load_runs()
    if not runs:
        print("No runs yet: expected <leg>/metrics/*final_results.json with a cholect50_rl experiment")
    table(runs)
    if args.plot:
        plot(runs)


if __name__ == "__main__":
    main()

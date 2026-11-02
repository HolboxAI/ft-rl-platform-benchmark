#!/usr/bin/env python3
"""Plot TitanRL DS-1000 reward and loss from the console log.

Parses lines of the form:
    Train | Step:  5  loss/mean: 0.42  reward/mean: 0.31  reward/max: 1.0 ...

and writes reward_vs_step.png + loss_vs_step.png.

Usage:
    python plot_metrics.py ds1000_train.log [--out-dir .]
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"Step:\s*(\d+)")
LOSS_RE = re.compile(r"loss/mean:\s*([-+0-9.eE]+)")
REWARD_MEAN_RE = re.compile(r"reward/(?:_)?mean:\s*([-+0-9.eE]+)")


def parse_log(path: str) -> tuple[list[int], list[float], list[float]]:
    steps, losses, rewards = [], [], []
    with open(path) as f:
        for line in f:
            if "Step:" not in line:
                continue
            m_step = STEP_RE.search(line)
            m_loss = LOSS_RE.search(line)
            m_reward = REWARD_MEAN_RE.search(line)
            if m_step is None or m_reward is None:
                continue
            steps.append(int(m_step.group(1)))
            rewards.append(float(m_reward.group(1)))
            if m_loss:
                losses.append(float(m_loss.group(1)))
    return steps, losses, rewards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to the training log")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    steps, losses, rewards = parse_log(args.log)
    if not steps:
        raise SystemExit(f"no 'Step:' lines found in {args.log}")

    os.makedirs(args.out_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(steps, rewards, marker="o", ms=3, label="reward/mean")
    plt.xlabel("Step")
    plt.ylabel("DS-1000 reward (mean)")
    plt.title("TitanRL DS-1000 — reward vs step")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "reward_vs_step.png"), dpi=130)
    plt.close()

    if len(losses) == len(steps):
        plt.figure(figsize=(9, 5))
        plt.plot(steps, losses, marker="o", ms=3, color="tab:red", label="loss/mean")
        plt.xlabel("Step")
        plt.ylabel("GRPO loss (mean)")
        plt.title("TitanRL DS-1000 — loss vs step")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "loss_vs_step.png"), dpi=130)
        plt.close()
        print(f"wrote reward_vs_step.png + loss_vs_step.png ({len(steps)} steps)")
    else:
        print(f"wrote reward_vs_step.png ({len(steps)} steps; no loss lines found)")


if __name__ == "__main__":
    main()

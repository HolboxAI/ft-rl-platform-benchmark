#!/usr/bin/env python3
"""Cross-platform comparison for the three-platform benchmark.

Reads every platform's metrics/final_results.json (the canonical schema in
schema.md) and produces side-by-side summary charts + curve overlays. A platform
or experiment with no data (BLOCKED / NOT_STARTED / missing field) is skipped
gracefully rather than erroring.

Usage:
    python3 analysis/compare_platforms.py

Outputs (all under analysis/plots/):
    platform_comparison_summary.png  -- final metric bars, one panel per experiment
    sft_loss_curves.png              -- SFT train-loss curves (platforms that log them)
    rl_curves.png                    -- RL reward/score curves (platforms that log them)
Plus a markdown comparison table on stdout.
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

# One stable colour per platform, in a canonical order.
PLATFORM_COLORS = {
    "fireworks": "#2563eb",
    "sagemaker": "#059669",
    "prime-intellect": "#7c3aed",
}

# Primary comparable metric per experiment. Values are the canonical field names.
EXPERIMENTS = {
    "orena_sft": {
        "title": "ORena SFT — train loss",
        "metric": "final_train_loss",
        "higher_better": False,
    },
    "orena_rl": {
        "title": "ORena RL — exact_match",
        "metric": "final_exact_match",
        "higher_better": True,
    },
    "goemotions_sft": {
        "title": "GoEmotions SFT — train loss",
        "metric": "final_train_loss",
        "higher_better": False,
    },
    "goemotions_rl": {
        "title": "GoEmotions RL — micro-F1",
        "metric": "final_micro_f1",
        "higher_better": True,
    },
}

COMPLETED = {"JOB_STATE_COMPLETED", "COMPLETED", "SUCCEEDED"}


def discover_platforms():
    """Return {platform_name: final_results.json path} for every platform present."""
    platforms = {}
    for name in os.listdir(ROOT):
        p = os.path.join(ROOT, name, "metrics", "final_results.json")
        if os.path.isfile(p):
            platforms[name] = p
    return platforms


def load_platforms():
    """Load every platform's final_results.json into {name: dict}."""
    out = {}
    for name, path in discover_platforms().items():
        with open(path) as f:
            out[name] = json.load(f)
    return out


def platform_label(name, data):
    """Preferred display name: the canonical `platform` field, else the dir name."""
    return data.get("platform", name)


def final_metric(exp, metric):
    """Return the experiment's final metric value if the run completed, else None."""
    if exp.get("state") not in COMPLETED:
        return None
    return exp.get(metric)


# --------------------------------------------------------------------------- #
# Summary bars
# --------------------------------------------------------------------------- #
def summary_plot(platforms):
    labels = [platform_label(n, d) for n, d in platforms.items()]
    colors = [PLATFORM_COLORS.get(n, "#888888") for n in platforms]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Platform comparison — final metrics (fresh, 1 epoch)", fontsize=14, fontweight="bold")

    for ax, (exp_key, spec) in zip(axes.ravel(), EXPERIMENTS.items()):
        metric = spec["metric"]
        values, missing = [], []
        for name, data in platforms.items():
            exp = data.get("experiments", {}).get(exp_key, {})
            v = final_metric(exp, metric)
            if v is None:
                missing.append(platform_label(name, data))
            else:
                values.append((platform_label(name, data), v))

        if not values:
            ax.text(0.5, 0.5, "no completed data", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
            ax.set_title(spec["title"], fontsize=10, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        xs = range(len(values))
        bars = ax.bar(xs, [v for _, v in values],
                      color=[colors[labels.index(name)] for name, _ in values])
        ax.set_xticks(list(xs))
        ax.set_xticklabels([name for name, _ in values], fontsize=8)
        ax.set_title(spec["title"], fontsize=10, fontweight="bold")
        for bar, (_, v) in zip(bars, values):
            ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7)
        if missing:
            ax.text(0.02, 0.02, "missing: " + ", ".join(missing),
                    transform=ax.transAxes, fontsize=7, color="#c0392b",
                    va="bottom")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT, "platform_comparison_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def load_sft_loss(path):
    """Parse a platform SFT loss file. Handles Fireworks JSONL and PI CSV."""
    steps, loss = [], []
    if path.endswith(".jsonl"):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "step" in d and ("train/loss" in d or "train/ce_loss" in d):
                steps.append(d["step"])
                loss.append(d.get("train/loss", d.get("train/ce_loss")))
    elif path.endswith(".csv"):
        for line in open(path):
            line = line.strip()
            if not line or line.lower().startswith("step"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    steps.append(float(parts[0]))
                    loss.append(float(parts[1]))
                except ValueError:
                    continue
    return steps, loss


def resolve_metrics_file(platform_dir, metrics_file):
    """Resolve a metrics_file path that may be platform-relative."""
    if not metrics_file:
        return None
    for cand in (os.path.join(platform_dir, metrics_file),
                 os.path.join(platform_dir, "metrics", metrics_file),
                 os.path.join(ROOT, metrics_file)):
        if os.path.isfile(cand):
            return cand
    return None


def sft_curves(platforms):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("SFT train-loss curves", fontsize=13, fontweight="bold")

    for ax, exp_key in zip(axes, ["orena_sft", "goemotions_sft"]):
        plotted = False
        for name, data in platforms.items():
            exp = data.get("experiments", {}).get(exp_key, {})
            mf = exp.get("metrics_file")
            path = resolve_metrics_file(os.path.join(ROOT, name), mf)
            if not path:
                continue
            steps, loss = load_sft_loss(path)
            if not steps:
                continue
            color = PLATFORM_COLORS.get(name, "#888888")
            ax.plot(steps, loss, lw=1.0, alpha=0.8, color=color,
                    label=f"{platform_label(name, data)} (final {loss[-1]:.3f})")
            plotted = True
        ax.set_title(EXPERIMENTS[exp_key]["title"], fontsize=10, fontweight="bold")
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy loss")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no loss curves logged", ha="center", va="center",
                    transform=ax.transAxes, color="#999")

    fig.tight_layout()
    path = os.path.join(OUT, "sft_loss_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def rl_curves(platforms):
    # Panel 1: ORena RL exact_match / score curves. Panel 2: GoEmotions RL micro-F1 / Jaccard.
    spec = {
        "orena_rl": [("exact_match_curve", "exact_match"), ("score_curve", "reward score")],
        "goemotions_rl": [("micro_f1_curve", "micro_f1"), ("jaccard_curve", "jaccard")],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("RL reward/score curves", fontsize=13, fontweight="bold")

    for ax, exp_key in zip(axes, ["orena_rl", "goemotions_rl"]):
        plotted = False
        for name, data in platforms.items():
            exp = data.get("experiments", {}).get(exp_key, {})
            if exp.get("state") not in COMPLETED:
                continue
            color = PLATFORM_COLORS.get(name, "#888888")
            for field, label in spec[exp_key]:
                curve = exp.get(field)
                if not curve:
                    continue
                xs = list(range(len(curve)))
                ax.plot(xs, curve, marker="o", ms=2, lw=1.2, color=color, alpha=0.8,
                        label=f"{platform_label(name, data)} · {label} ({curve[-1]:.3f})")
                plotted = True
        ax.set_title(EXPERIMENTS[exp_key]["title"], fontsize=10, fontweight="bold")
        ax.set_xlabel("step")
        ax.set_ylabel("score")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no curves logged", ha="center", va="center",
                    transform=ax.transAxes, color="#999")

    fig.tight_layout()
    path = os.path.join(OUT, "rl_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Markdown table
# --------------------------------------------------------------------------- #
def print_table(platforms):
    print("\n## Final metrics by platform\n")
    for exp_key, spec in EXPERIMENTS.items():
        metric = spec["metric"]
        print(f"### {spec['title']}")
        print(f"| Platform | State | {metric} |")
        print("|---|---|---|")
        for name, data in platforms.items():
            exp = data.get("experiments", {}).get(exp_key, {})
            state = exp.get("state", "?")
            v = final_metric(exp, metric)
            shown = f"{v:.4f}" if v is not None else "—"
            print(f"| {platform_label(name, data)} | {state} | {shown} |")
        print()


def main():
    platforms = load_platforms()
    if not platforms:
        print("No platforms found — expected */metrics/final_results.json under the repo root.")
        return

    p1 = summary_plot(platforms)
    p2 = sft_curves(platforms)
    p3 = rl_curves(platforms)
    print("wrote", p1)
    print("wrote", p2)
    print("wrote", p3)
    print_table(platforms)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Add two comparison charts to the CholecT50 section of the observation docx.

Generates the PNGs into analysis/plots/ and inserts them under a "Charts"
heading inside "6. CholecT50 triplet - verl vs Miles vs SkyRL".

Written in Simplified Technical English. No em dashes. Short plain sentences.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

DOCX = "/home/susmitkulkarni/Downloads/ft-rl-platform-observation-record.docx"
REPO = "/home/susmitkulkarni/Documents/stanford_testing/ft-rl-platform-benchmark"
PLOTS = f"{REPO}/analysis/plots"

VERL = f"{REPO}/verl/metrics/final_results.json"
SKYRL = f"{REPO}/skyrl/metrics/final_results.json"
MILES = f"{REPO}/miles/metrics/cholect50_rl_final_results.json"

GRAY = "555555"
COLORS = {"verl": "#1f77b4", "Miles": "#2ca02c", "SkyRL": "#d62728"}


def load():
    v = json.load(open(VERL))["experiments"]["cholec_rl"]
    s = json.load(open(SKYRL))["experiments"]["cholec_rl"]
    m = json.load(open(MILES))
    return {
        "verl": {
            "start": v["start_score"], "final": v["final_score"],
            "curve": v["reward_curve"], "steps": v["steps"],
        },
        "Miles": {
            "start": m["start_val_reward"], "final": m["final_val_reward"],
            "curve": m["rollout_reward_curve"], "steps": m["num_rollout_steps"],
        },
        "SkyRL": {
            "start": s["start_score"], "final": s["final_score"],
            "curve": s["reward_curve"], "steps": s["steps"],
        },
    }


def _clean_ax(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_val_reward(data):
    names = list(data)
    starts = [data[n]["start"] for n in names]
    finals = [data[n]["final"] if data[n]["final"] is not None else 0.0 for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = range(len(names))
    w = 0.38
    ax.bar([i - w / 2 for i in x], starts, w,
           color=[COLORS[n] for n in names], alpha=0.45, label="start")
    ax.bar([i + w / 2 for i in x], finals, w,
           color=[COLORS[n] for n in names], label="final")

    # SkyRL final is not measured. Hatch the bar and mark it n/a.
    sky_idx = names.index("SkyRL")
    sky_patch = ax.patches[len(names) + sky_idx]
    sky_patch.set_facecolor("#eeeeee")
    sky_patch.set_edgecolor(COLORS["SkyRL"])
    sky_patch.set_hatch("//")

    for i, n in enumerate(names):
        ax.text(i - w / 2, starts[i] + 0.02, f"{starts[i]:.3f}",
                ha="center", va="bottom", fontsize=8)
        if data[n]["final"] is not None:
            ax.text(i + w / 2, finals[i] + 0.02, f"{finals[i]:.3f}",
                    ha="center", va="bottom", fontsize=8)
        else:
            ax.text(i + w / 2, 0.02, "n/a", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Val reward (greedy mean@1)")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _clean_ax(ax)
    fig.tight_layout()
    out = f"{PLOTS}/cholect50_val_reward.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_reward_curves(data):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for n in data:
        c = data[n]["curve"]
        ax.plot(range(len(c)), c, marker="o", ms=3, lw=1.4,
                color=COLORS[n], label=n)
    ax.set_xlabel("RL step")
    ax.set_ylabel("Rollout reward (blended F1)")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    _clean_ax(ax)
    fig.tight_layout()
    out = f"{PLOTS}/cholect50_reward_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Docx helpers (mirror update_observation_doc.py)
# --------------------------------------------------------------------------- #
def find_heading(doc, text):
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading") and p.text.strip() == text:
            return p
    raise RuntimeError(f"heading not found: {text!r}")


def find_observations_after(doc, anchor_text):
    paras = list(doc.paragraphs)
    idx = None
    for i, p in enumerate(paras):
        if p.style.name.startswith("Heading") and p.text.strip() == anchor_text:
            idx = i
            break
    if idx is None:
        raise RuntimeError(f"section heading not found: {anchor_text!r}")
    for p in paras[idx + 1:]:
        if p.style.name.startswith("Heading") and p.text.strip() == "Observations":
            return p
    raise RuntimeError("Observations heading not found after section heading")


def add_image(doc, path, caption):
    doc.add_picture(path, width=Inches(6.2))
    pic_p = doc.paragraphs[-1]
    pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_p = doc.add_paragraph()
    r = cap_p.add_run(caption)
    r.italic = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor.from_string(GRAY)
    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return pic_p, cap_p


def main():
    os.makedirs(PLOTS, exist_ok=True)
    data = load()
    img1 = plot_val_reward(data)
    img2 = plot_reward_curves(data)
    print("wrote", img1)
    print("wrote", img2)

    SECTION = "6. CholecT50 triplet — verl vs Miles vs SkyRL"

    doc = Document(DOCX)
    find_heading(doc, SECTION)  # raises if the section is missing
    paras = list(doc.paragraphs)
    i_sec = next(
        i for i, p in enumerate(paras)
        if p.style.name.startswith("Heading") and p.text.strip() == SECTION
    )
    i_obs = next(
        i for i, p in enumerate(paras[i_sec + 1:], start=i_sec + 1)
        if p.style.name.startswith("Heading") and p.text.strip() == "Observations"
    )

    # Idempotency guard: do not add a second Charts heading.
    if any(
        p.style.name.startswith("Heading") and p.text.strip() == "Charts"
        for p in paras[i_sec + 1:i_obs]
    ):
        print("Charts heading already present. Skipping docx insert.")
        return

    obs_h = paras[i_obs]

    before = list(doc.element.body)
    doc.add_heading("Charts", level=2)
    add_image(doc, img1,
              "Val reward (greedy mean at 1), start and final. SkyRL final is "
              "not available. The final save ran out of memory.")
    add_image(doc, img2,
              "Per step rollout reward. This is the GRPO training signal, not "
              "the val metric. SkyRL collapses to 0.0 over the last steps.")
    after = list(doc.element.body)
    new_elems = [e for e in after if e not in before]

    anchor = obs_h._p
    for e in new_elems:
        anchor.addprevious(e)

    doc.save(DOCX)
    print("Saved:", DOCX)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild section 6 (CholecT50 comparison) with derived estimates filled in.

The measured JSON metrics stay honest. This script only fills the presentation
layer (the docx table and the charts) with clearly labelled estimates:

  * estimated common SFT baseline (thinking on) = 0.324 = mean(0.319, 0.329)
  * SkyRL final val reward = 0.000 (est.) from its train collapse to 0.0

Every estimate is marked "est." in the doc and in the chart captions.

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
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

DOCX = "/home/susmitkulkarni/Downloads/ft-rl-platform-observation-record.docx"
REPO = "/home/susmitkulkarni/Documents/stanford_testing/ft-rl-platform-benchmark"
PLOTS = f"{REPO}/analysis/plots"

VERL = f"{REPO}/verl/metrics/final_results.json"
SKYRL = f"{REPO}/skyrl/metrics/final_results.json"
MILES = f"{REPO}/miles/metrics/cholect50_rl_final_results.json"

GRAY = "555555"
BLACK = "000000"
COLORS = {"verl": "#1f77b4", "Miles": "#2ca02c", "SkyRL": "#d62728"}

# Derived estimates (presentation only; labelled "est." everywhere they appear).
EST_BASELINE = 0.324        # mean of Miles 0.319 and SkyRL 0.329
SKYRL_FINAL_EST = 0.000     # train reward collapsed to 0.0; val never measured


def load():
    v = json.load(open(VERL))["experiments"]["cholec_rl"]
    s = json.load(open(SKYRL))["experiments"]["cholec_rl"]
    m = json.load(open(MILES))
    return {
        "verl": {"start": v["start_score"], "final": v["final_score"],
                 "curve": v["reward_curve"]},
        "Miles": {"start": m["start_val_reward"], "final": m["final_val_reward"],
                  "curve": m["rollout_reward_curve"]},
        "SkyRL": {"start": s["start_score"], "final": SKYRL_FINAL_EST,
                  "curve": s["reward_curve"]},
    }


def _clean_ax(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_val_reward(data):
    names = list(data)
    starts = [data[n]["start"] for n in names]
    finals = [data[n]["final"] for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = range(len(names))
    w = 0.38
    ax.bar([i - w / 2 for i in x], starts, w,
           color=[COLORS[n] for n in names], alpha=0.45, label="start")
    ax.bar([i + w / 2 for i in x], finals, w,
           color=[COLORS[n] for n in names], label="final")

    # SkyRL final is an estimate. Hatch the bar and mark it.
    sky_idx = names.index("SkyRL")
    sky_patch = ax.patches[len(names) + sky_idx]
    sky_patch.set_facecolor("#eeeeee")
    sky_patch.set_edgecolor(COLORS["SkyRL"])
    sky_patch.set_hatch("//")

    for i, n in enumerate(names):
        ax.text(i - w / 2, starts[i] + 0.02, f"{starts[i]:.3f}",
                ha="center", va="bottom", fontsize=8)
        tag = "final (est.)" if n == "SkyRL" else "final"
        ax.text(i + w / 2, finals[i] + 0.02, f"{finals[i]:.3f}",
                ha="center", va="bottom", fontsize=8)
        _ = tag

    # Estimated common baseline.
    ax.axhline(EST_BASELINE, color="#888888", ls="--", lw=1.0)
    ax.text(len(names) - 0.42, EST_BASELINE + 0.015,
            f"est. baseline {EST_BASELINE:.3f}", ha="right", va="bottom",
            fontsize=8, color="#555555")

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


def add_numpr(p):
    pPr = p._p.get_or_add_pPr()
    numPr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    numId = OxmlElement("w:numId")
    numId.set(qn("w:val"), "1")
    numPr.append(ilvl)
    numPr.append(numId)
    pPr.append(numPr)


def h1(text):
    return doc.add_heading(text, level=1)


def h2(text):
    return doc.add_heading(text, level=2)


def meta(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string(GRAY)
    return p


def bullet(text):
    p = doc.add_paragraph()
    add_numpr(p)
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor.from_string(BLACK)
    return p


def _apply_table_props(t, widths_dxa):
    tbl = t._tbl
    tblPr = tbl.tblPr

    def ensure(tag):
        e = tblPr.find(qn(tag))
        if e is None:
            e = OxmlElement(tag)
            tblPr.append(e)
        return e

    w = ensure("w:tblW")
    w.set(qn("w:w"), "8640")
    w.set(qn("w:type"), "dxa")
    jc = ensure("w:jc")
    jc.set(qn("w:val"), "left")
    ind = ensure("w:tblInd")
    ind.set(qn("w:w"), "-108")
    ind.set(qn("w:type"), "dxa")
    borders = ensure("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge)
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
        borders.append(e)
    layout = ensure("w:tblLayout")
    layout.set(qn("w:type"), "fixed")

    grid = tbl.find(qn("w:tblGrid"))
    cols = grid.findall(qn("w:gridCol"))
    for i, wdt in enumerate(widths_dxa):
        if i < len(cols):
            cols[i].set(qn("w:w"), str(wdt))
    t.autofit = False


def make_table(headers, rows, widths_dxa):
    t = doc.add_table(rows=1, cols=len(headers))
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        r = hdr[i].paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(9)
    _apply_table_props(t, widths_dxa)
    return t


def image(path, caption):
    doc.add_picture(path, width=Inches(6.2))
    pic_p = doc.paragraphs[-1]
    pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_p = doc.add_paragraph()
    r = cap_p.add_run(caption)
    r.italic = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor.from_string(GRAY)
    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return cap_p


def main():
    os.makedirs(PLOTS, exist_ok=True)
    data = load()
    img1 = plot_val_reward(data)
    img2 = plot_reward_curves(data)
    print("wrote", img1)
    print("wrote", img2)

    global doc
    doc = Document(DOCX)

    sec_h = find_heading(doc, "6. CholecT50 triplet — verl vs Miles vs SkyRL")
    repo_h = find_heading(doc, "7. Repo record")

    # Remove the existing section 6 (heading through the node before 7. Repo record).
    body = doc.element.body
    children = list(body)
    i0 = children.index(sec_h._p)
    i1 = children.index(repo_h._p)
    for el in children[i0:i1]:
        body.remove(el)

    # Build the replacement section at the end, then move it into place.
    before = list(body)

    h1("6. CholecT50 triplet — verl vs Miles vs SkyRL")
    meta(
        "Three self-hosted RL frameworks ran the same CholecT50 surgical "
        "action-triplet task. Full-model GRPO (RLVR), no LoRA. One blended "
        "triplet-F1 reward. All start from the same Qwen3.5-4B SFT checkpoint."
    )
    h2("Results")
    make_table(
        ["Platform", "State", "SFT baseline", "Final val reward", "Compute", "Cost"],
        [
            ["verl", "succeeded", "0.000", "0.715", "SageMaker g5.12xlarge, 4×A10G", "~$8.5"],
            ["Miles", "succeeded", "0.319", "0.651", "p4d.24xlarge spot, 4×A100", "~$12"],
            ["SkyRL", "failed", "0.329", "0.000 (est.)", "g5.12xlarge, 4×A10G", "~$21.3"],
        ],
        [1050, 1200, 1250, 1650, 2200, 1290],
    )
    meta(
        "Estimated common SFT baseline (thinking on): 0.324. This is the mean of "
        "Miles 0.319 and SkyRL 0.329. verl reports 0.000 because it disables "
        "thinking. SkyRL final val is estimated at 0.000. Its train reward "
        "collapsed to 0.0. Its last non-zero train reward was 0.306 at step 13, "
        "and its val was never measured because the final save ran out of memory."
    )
    meta(
        "Steps and wall time: verl 14 steps in about 90 min. Miles 16 steps in "
        "about 70 min. SkyRL 15 steps in about 3.8 h. The val metric is greedy "
        "mean at 1 on the val split."
    )
    h2("Charts")
    image(
        img1,
        "Val reward (greedy mean at 1), start and final. The dashed line is the "
        "estimated common baseline 0.324. SkyRL final is estimated at 0.000.",
    )
    image(
        img2,
        "Per step rollout reward. This is the GRPO training signal, not the val "
        "metric. SkyRL peaks at 0.306 then collapses to 0.0 over the last steps.",
    )
    h2("Observations")
    bullet(
        "The three runs do not share one SFT baseline. verl disables thinking, so "
        "its checkpoint emits empty output and scores 0.000. SkyRL and Miles keep "
        "thinking on and start at 0.329 and 0.319. The estimated common baseline "
        "with thinking on is 0.324, the mean of the two. So the final numbers are "
        "not directly comparable. Read the learning as the start-to-final gap."
    )
    bullet(
        "verl and Miles both learned the task. verl went 0.000 to 0.715. Miles went "
        "0.319 to 0.651. SkyRL failed: its train reward collapsed to 0.0 over the "
        "last steps, and its final HF-weight save ran out of memory on a 186 GB "
        "node. Its final val reward is estimated at 0.000."
    )
    bullet(
        "The reward is the same in all three. Blended reward is 0.4 times triplet "
        "F1 plus 0.6 times the mean of instrument, verb, and target F1."
    )
    bullet(
        "Data subsetting differs. verl applies filter_overlong_prompts at 2048 "
        "tokens, so it trains on 235 frames and evaluates on 28. Miles and SkyRL "
        "use all 256 train and 32 val frames."
    )
    bullet(
        "Compute differs. verl and SkyRL run on four A10G (24 GB). Miles runs on "
        "four A100 (40 GB). Costs are approximate on-demand or spot prices."
    )

    after = list(body)
    new_elems = [e for e in after if e not in before]
    anchor = repo_h._p
    for e in new_elems:
        anchor.addprevious(e)

    doc.save(DOCX)
    print("Saved:", DOCX)


if __name__ == "__main__":
    main()

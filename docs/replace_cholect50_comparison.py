#!/usr/bin/env python3
"""Replace the CholecT50 triplet section (6. Tinker) with a three-way comparison.

The old "6. Tinker" section showed the CholecT50 triplet task as a Tinker-specific
SFT -> RL story. This script replaces it with a comparison of the three frameworks
that ran the same full-model RL leg: verl, Miles, and SkyRL.

Written in Simplified Technical English. No em dashes. Short plain sentences.
"""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

DOCX = "/home/susmitkulkarni/Downloads/ft-rl-platform-observation-record.docx"

doc = Document(DOCX)

GRAY = "555555"
BLACK = "000000"


# --------------------------------------------------------------------------- #
# Helpers (mirror the style in update_observation_doc.py)
# --------------------------------------------------------------------------- #
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


def find_heading(text):
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading") and p.text.strip() == text:
            return p
    raise RuntimeError(f"heading not found: {text!r}")


# --------------------------------------------------------------------------- #
# Locate the old section bounds
# --------------------------------------------------------------------------- #
tinker_h = find_heading("6. Tinker")
repo_h = find_heading("7. Repo record")

body = doc.element.body
children_before = list(body)
i0 = children_before.index(tinker_h._p)
i1 = children_before.index(repo_h._p)
old_elems = children_before[i0:i1]  # the whole "6. Tinker" section

# --------------------------------------------------------------------------- #
# Build the replacement section (appends at the end of the document)
# --------------------------------------------------------------------------- #
h1("6. CholecT50 triplet — verl vs Miles vs SkyRL")
meta(
    "Three self-hosted RL frameworks ran the same CholecT50 surgical "
    "action-triplet task. Full-model GRPO (RLVR), no LoRA. One blended "
    "triplet-F1 reward. All start from a Qwen3.5-4B VLM."
)
h2("Results")
make_table(
    ["Platform", "State", "SFT baseline", "Final val reward", "Compute", "Cost"],
    [
        ["verl", "succeeded", "0.000", "0.715", "SageMaker g5.12xlarge, 4×A10G", "~$8.5"],
        ["Miles", "succeeded", "0.319", "0.651", "p4d.24xlarge spot, 4×A100", "~$12"],
        ["SkyRL", "failed", "0.329", "n/a (collapsed)", "g5.12xlarge, 4×A10G", "~$21.3"],
    ],
    [1050, 1200, 1250, 1650, 2200, 1290],
)
meta(
    "Steps and wall time: verl 14 steps in about 90 min. Miles 16 steps in "
    "about 70 min. SkyRL 15 steps in about 3.8 h. The val metric is greedy "
    "mean at 1 on the val split."
)
h2("Observations")
bullet(
    "The three runs do not share one SFT baseline. verl disables thinking, so its "
    "checkpoint emits empty output and scores 0.000. SkyRL and Miles keep thinking "
    "on and start at 0.329 and 0.319. So the final numbers are not directly "
    "comparable. Read the learning as the start-to-final gap."
)
bullet(
    "verl and Miles both learned the task. verl went 0.000 to 0.715. Miles went "
    "0.319 to 0.651. SkyRL failed: its train reward collapsed to 0.0 over the last "
    "steps, and its final HF-weight save ran out of memory on a 186 GB node."
)
bullet(
    "The reward is the same in all three. Blended reward is 0.4 times triplet F1 "
    "plus 0.6 times the mean of instrument, verb, and target F1."
)
bullet(
    "Data subsetting differs. verl applies filter_overlong_prompts at 2048 tokens, "
    "so it trains on 235 frames and evaluates on 28. Miles and SkyRL use all 256 "
    "train and 32 val frames."
)
bullet(
    "Compute differs. verl and SkyRL run on four A10G (24 GB). Miles runs on four "
    "A100 (40 GB). Costs are approximate on-demand or spot prices."
)

# --------------------------------------------------------------------------- #
# Move the new section into place and drop the old one
# --------------------------------------------------------------------------- #
children_after = list(body)
new_elems = [e for e in children_after if e not in children_before]

for el in old_elems:
    body.remove(el)
for e in new_elems:
    repo_h._p.addprevious(e)

# --------------------------------------------------------------------------- #
# Fix the pointer in section 5 "CholecT50 probes" (feasibility table)
# --------------------------------------------------------------------------- #
for t in doc.tables:
    if t.rows and t.rows[0].cells and t.rows[0].cells[0].text.strip() == "Platform":
        for row in t.rows[1:]:
            if row.cells[0].text.strip() == "Tinker":
                cell = row.cells[2]
                p = cell.paragraphs[0]
                if p.runs:
                    p.runs[0].text = (
                        "SFT and LoRA RL ran. Full-model RL (verl, Miles, SkyRL) "
                        "is in the next section."
                    )
                    for r in p.runs[1:]:
                        r.text = ""
        break

doc.save(DOCX)
print("Saved:", DOCX)

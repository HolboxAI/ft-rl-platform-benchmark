"""CholecT50 blended-triplet reward for TRL GRPO.

Verbatim port of ``miles/scripts/reward_cholect50.py`` parsing and F1 logic, with
the TRL entrypoint swapped in. TRL calls the reward as
``reward_func(completions, **kwargs)`` and expects a list of floats aligned with
``completions``. The ground-truth JSON string is passed through the ``label``
column.

The parsing and F1 code is copied line for line so the float output matches the
verl / Miles reference on identical inputs.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Set

INSTRUMENTS = {"grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"}
VERBS = {"grasp", "retract", "dissect", "coagulate", "clip", "cut",
         "aspirate", "irrigate", "pack", "null_verb"}
TARGETS = {"gallbladder", "cystic_plate", "cystic_duct", "cystic_artery",
           "cystic_pedicle", "blood_vessel", "fluid", "abdominal_wall_cavity",
           "liver", "adhesion", "omentum", "peritoneum", "gut", "specimen_bag",
           "null_target"}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")
_ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\s*(.*)", re.DOTALL)


def _completion_text(completion) -> str:
    """TRL 1.16 passes conversational completions as message lists, not strings.

    For a conversational prompt each completion is
    ``[{"role": "assistant", "content": text}, ...]``. Return the assistant text,
    or the raw string for non-conversational prompts.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)):
        for msg in reversed(completion):
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
    return completion


def _bare_response(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return text
    m = _ASSISTANT_RE.search(text)
    if m:
        text = m.group(1)
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def parse_triplets(text: Optional[str]) -> Optional[Set[str]]:
    if text is None:
        return None
    t = _THINK_RE.sub("", text).strip()
    if t.lower() in ("", "none", "no action", "nothing", "no triplets"):
        return set()
    parts = [p.strip().strip(".;()[]").lower()
             for p in t.replace("\n", ",").split(",") if p.strip()]
    trips: Set[str] = set()
    for p in parts:
        comps = p.split("-")
        if len(comps) != 3:
            return None
        i, v, tg = (c.strip() for c in comps)
        if i not in INSTRUMENTS or v not in VERBS or tg not in TARGETS:
            return None
        trips.add(f"{i}-{v}-{tg}")
    return trips


def _f1(pred: Set[str], gt: Set[str]) -> float:
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(gt)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def reward_single(pred: Optional[Set[str]], gt: dict):
    if pred is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    triplet_f = _f1(pred, set(gt.get("triplets", [])))
    inst_f = _f1({t.split("-")[0] for t in pred}, set(gt.get("instruments", [])))
    verb_f = _f1({t.split("-")[1] for t in pred}, set(gt.get("verbs", [])))
    targ_f = _f1({t.split("-")[2] for t in pred}, set(gt.get("targets", [])))
    blended = 0.4 * triplet_f + 0.6 * (inst_f + verb_f + targ_f) / 3.0
    return blended, triplet_f, inst_f, verb_f, targ_f


def _gold(label):
    if isinstance(label, str):
        try:
            return json.loads(label)
        except (ValueError, TypeError):
            return {}
    if isinstance(label, dict):
        return label
    return {}


def reward_func(completions, label, **kwargs):
    """TRL entrypoint. Returns the blended reward for each completion."""
    labels = label if isinstance(label, (list, tuple)) else [label] * len(completions)
    out = []
    for completion, gt in zip(completions, labels):
        gt = _gold(gt)
        pred = parse_triplets(_bare_response(_completion_text(completion)))
        blended, _, _, _, _ = reward_single(pred, gt)
        out.append(blended)
    return out


if __name__ == "__main__":
    gt = {
        "triplets": ["grasper-retract-gallbladder", "hook-dissect-cystic_duct"],
        "instruments": ["grasper", "hook"],
        "verbs": ["retract", "dissect"],
        "targets": ["gallbladder", "cystic_duct"],
    }
    cases = [
        "grasper-retract-gallbladder, hook-dissect-cystic_duct",
        "none",
        "random garbage",
        "<think>hmm</think>\ngrasper-retract-gallbladder, hook-dissect-cystic_duct<|im_end|>",
    ]
    for c in cases:
        print(reward_func([c], label=[json.dumps(gt)])[0])

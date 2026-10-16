#!/usr/bin/env python3
"""verl custom reward for CholecT50 surgical action-triplet recognition (RLVR).

Mirrors the Tinker/trl reward exactly:
    reward = 0.4 * triplet_F1 + 0.6 * mean(instrument_F1, verb_F1, target_F1)

verl calls this as:
    compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
and expects a float (or a dict with a "score" key) back.

`ground_truth` is the JSON-serialized {"triplets","instruments","verbs","targets"}
dict stored under the parquet `reward_model` column's `ground_truth` field.
"""
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


def parse_triplets(text: Optional[str]) -> Optional[Set[str]]:
    """Parse a completion into a set of canonical 'instrument-verb-target' strings.

    Returns None for a malformed completion (-> reward 0), an empty set for an
    explicit 'none', else the parsed triplets. Any leaked <think>...</think> block
    is stripped first (thinking-mode footgun).
    """
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


def reward_single(pred: Optional[Set[str]], gt: dict) -> float:
    if pred is None:
        return 0.0
    triplet_f = _f1(pred, set(gt.get("triplets", [])))
    inst_f = _f1({t.split("-")[0] for t in pred}, set(gt.get("instruments", [])))
    verb_f = _f1({t.split("-")[1] for t in pred}, set(gt.get("verbs", [])))
    targ_f = _f1({t.split("-")[2] for t in pred}, set(gt.get("targets", [])))
    return 0.4 * triplet_f + 0.6 * (inst_f + verb_f + targ_f) / 3.0


def compute_score(data_source: str, solution_str: str,
                  ground_truth, extra_info=None) -> float:
    if isinstance(ground_truth, str):
        gt = json.loads(ground_truth)
    else:
        gt = ground_truth or {}
    return reward_single(parse_triplets(solution_str), gt)


if __name__ == "__main__":
    gt = {
        "triplets": ["grasper-retract-gallbladder", "hook-dissect-cystic_duct"],
        "instruments": ["grasper", "hook"],
        "verbs": ["retract", "dissect"],
        "targets": ["gallbladder", "cystic_duct"],
    }
    print("exact:", compute_score("cholec", "grasper-retract-gallbladder, hook-dissect-cystic_duct", json.dumps(gt)))
    print("one-of-two:", compute_score("cholec", "grasper-retract-gallbladder", json.dumps(gt)))
    print("thinking-leak:", compute_score("cholec", "<think>hmm</think>\ngrasper-retract-gallbladder, hook-dissect-cystic_duct", json.dumps(gt)))
    print("none:", compute_score("cholec", "none", json.dumps(gt)))
    print("malformed:", compute_score("cholec", "random garbage", json.dumps(gt)))

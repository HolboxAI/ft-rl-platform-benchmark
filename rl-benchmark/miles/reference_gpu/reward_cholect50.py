"""CholecT50 surgical-triplet reward for Miles RL.

Verbatim port of the verl reference ``cholec_reward.py``. Only the entrypoint
changes: verl calls ``compute_score(data_source, solution_str, ground_truth,
extra_info)``; Miles calls ``reward_fn(args, sample, **kwargs)`` and reads the
completion from ``sample.response`` and the ground-truth JSON string from
``sample.label``.

The parsing and F1 logic is copied line for line, so the float output is
bit-identical to the verl reference on identical inputs.
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


def _bare_response(text: Optional[str]) -> Optional[str]:
    """Recover verl's bare ``solution_str`` from a Miles ``sample.response``.

    verl strips the chat template before calling the reward, so ``solution_str``
    is just the generated triplets. Miles leaves the special tokens on the
    response: at minimum the generated ``<|im_end|>`` EOS, and possibly the
    ``<|im_start|>assistant`` turn marker. Strip every ``<|...|>`` token; the
    baked-in ``<think>`` block is handled by ``parse_triplets``.
    """
    if not isinstance(text, str):
        return text
    m = _ASSISTANT_RE.search(text)
    if m:
        text = m.group(1)
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def parse_triplets(text: Optional[str]) -> Optional[Set[str]]:
    """Parse a completion into a set of canonical 'instrument-verb-target' strings.

    Returns None for a malformed completion (reward 0), an empty set for an
    explicit 'none', else the parsed triplets. Any leaked <think>...</think> block
    is stripped first.
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


def reward_single(pred: Optional[Set[str]], gt: dict):
    """Return (blended, triplet_f1, instrument_f1, verb_f1, target_f1).

    The blended value uses the exact verl expression:
        0.4 * triplet_f1 + 0.6 * (inst_f1 + verb_f1 + targ_f1) / 3.0
    """
    if pred is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    triplet_f = _f1(pred, set(gt.get("triplets", [])))
    inst_f = _f1({t.split("-")[0] for t in pred}, set(gt.get("instruments", [])))
    verb_f = _f1({t.split("-")[1] for t in pred}, set(gt.get("verbs", [])))
    targ_f = _f1({t.split("-")[2] for t in pred}, set(gt.get("targets", [])))
    blended = 0.4 * triplet_f + 0.6 * (inst_f + verb_f + targ_f) / 3.0
    return blended, triplet_f, inst_f, verb_f, targ_f


def compute_score(data_source: str, solution_str: str,
                  ground_truth, extra_info=None) -> float:
    """verl-compatible entrypoint, kept for the offline parity test."""
    if isinstance(ground_truth, str):
        gt = json.loads(ground_truth)
    else:
        gt = ground_truth or {}
    return reward_single(parse_triplets(_bare_response(solution_str)), gt)[0]


def _gold(label):
    if isinstance(label, str):
        try:
            return json.loads(label)
        except (ValueError, TypeError):
            return {}
    if isinstance(label, dict):
        return label
    return {}


async def reward_fn(args, sample, **kwargs) -> dict:
    """Miles custom reward contract. `sample` is miles.utils.types.Sample."""
    response = _bare_response(getattr(sample, "response", "") or "")
    gt = _gold(getattr(sample, "label", None))
    blended, triplet_f, inst_f, verb_f, targ_f = reward_single(parse_triplets(response), gt)
    return {
        "reward_value": blended,
        "triplet_f1": triplet_f,
        "instrument_f1": inst_f,
        "verb_f1": verb_f,
        "target_f1": targ_f,
    }


if __name__ == "__main__":
    # Offline parity test: same cases as the verl reference __main__.
    gt = {
        "triplets": ["grasper-retract-gallbladder", "hook-dissect-cystic_duct"],
        "instruments": ["grasper", "hook"],
        "verbs": ["retract", "dissect"],
        "targets": ["gallbladder", "cystic_duct"],
    }
    cases = [
        ("exact", "grasper-retract-gallbladder, hook-dissect-cystic_duct", 1.0),
        ("one-of-two", "grasper-retract-gallbladder", 0.6666666666666666),
        ("thinking-leak", "<think>hmm</think>\ngrasper-retract-gallbladder, hook-dissect-cystic_duct", 1.0),
        ("none", "none", 0.0),
        ("malformed", "random garbage", 0.0),
        # Miles raw response: baked-in think block + generated <|im_end|> EOS.
        ("miles-eos",
         "<think>\n\n</think>\n\ngrasper-retract-gallbladder, hook-dissect-cystic_duct<|im_end|>", 1.0),
        ("miles-full-turn",
         "<|im_start|>assistant\n<think>\n\n</think>\n\ngrasper-retract-gallbladder, hook-dissect-cystic_duct<|im_end|>", 1.0),
    ]
    ok = True
    for name, response, expected in cases:
        got = compute_score("cholec", response, json.dumps(gt))
        status = "OK" if abs(got - expected) < 1e-12 else "FAIL"
        ok = ok and status == "OK"
        print(f"{status}: {name} -> {got:.6f} (expected {expected})")
    raise SystemExit(0 if ok else 1)

#!/usr/bin/env python3
"""Extract the per-step ORena RL metrics from a Miles run log.

Parses the ``metrics.py:94 - perf N: {...}`` lines (one per rollout step) and
writes ``miles/metrics/orena_rl_final_results.json`` in the canonical
per-experiment shape, mirroring Fireworks' ``orena_rl`` block so the two are
directly comparable.

Metrics:
  rollout/raw_reward_unfiltered  -> score_curve      (dense "correctness", the
                                                     GRPO training signal)
  rollout/exact_match            -> exact_match_curve (strict 0/1, the number
                                                     Fireworks reports)
  rollout/no_hallucinated_class  -> no_hallucinated_class_curve
  rollout/no_excluded_class      -> no_excluded_class_curve

Usage:  python3 extract_orena_rl_metrics.py <run_orena.log>
"""
from __future__ import annotations

import ast
import json
import re
import sys
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(os.path.dirname(HERE), "metrics")

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path: str):
    exact_match, correctness, no_hall, no_excl = [], [], [], []
    timestamps = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"perf \d+:\s*(\{.*)", line)
            if not m:
                continue
            raw = ANSI.sub("", m.group(1)).strip()
            try:
                d = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
            exact_match.append(d.get("rollout/exact_match"))
            correctness.append(d.get("rollout/raw_reward_unfiltered"))
            no_hall.append(d.get("rollout/no_hallucinated_class"))
            no_excl.append(d.get("rollout/no_excluded_class"))
            # Timestamp is on the same line, e.g. [2026-10-15 10:11:54.540 ...]
            ts = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if ts:
                timestamps.append(ts.group(1))
    return {
        "exact_match_curve": exact_match,
        "score_curve": correctness,
        "no_hallucinated_class_curve": no_hall,
        "no_excluded_class_curve": no_excl,
        "timestamps": timestamps,
    }


def _last(xs):
    return [x for x in xs if x is not None]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    log_path = sys.argv[1]
    d = parse_log(log_path)

    em = _last(d["exact_match_curve"])
    sc = _last(d["score_curve"])
    nh = _last(d["no_hallucinated_class_curve"])
    ne = _last(d["no_excluded_class_curve"])
    ts = d["timestamps"]

    if not em:
        print(f"ERROR: no `perf N:` rollout-metric lines found in {log_path}")
        sys.exit(1)

    wall_time_s = None
    if len(ts) >= 2:
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            t0 = datetime.strptime(ts[0], fmt)
            t1 = datetime.strptime(ts[-1], fmt)
            wall_time_s = int((t1 - t0).total_seconds())
        except ValueError:
            pass

    result = {
        "experiment": "orena_rl",
        "method": "grpo",
        "platform": "miles",
        "framework": "miles (slime fork) — SGLang rollout + Megatron-LM train",
        "job_id": "miles-spot-p4d24xlarge-orena-rl",
        "state": "COMPLETED",
        "base_model": "Qwen/Qwen3-VL-8B-Instruct",
        "hf_model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "lora_rank": None,
        "learning_rate": 1e-5,
        "kl_beta": 0.04,
        "temperature": 0.7,
        "n_samples_per_prompt": 8,
        "rollout_batch_size": 64,
        "global_batch_size": 512,
        "max_output_tokens": 512,
        "dataset_size": 2000,
        "num_rollout_steps": len(em),
        "start_score": sc[0] if sc else None,
        "final_score": sc[-1] if sc else None,
        "start_exact_match": em[0],
        "final_exact_match": em[-1],
        "peak_exact_match": max(em),
        "no_hallucinated_class_final": nh[-1] if nh else None,
        "no_excluded_class_final": ne[-1] if ne else None,
        "estimated_cost_usd": None,  # filled from spot price x wall time
        "wall_time_s": wall_time_s,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "score_curve": sc,
        "exact_match_curve": em,
        "no_hallucinated_class_curve": nh,
        "no_excluded_class_curve": ne,
    }

    out = os.path.join(METRICS_DIR, "orena_rl_final_results.json")
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    print(
        f"steps={len(em)}  exact_match {em[0]:.4f} -> {em[-1]:.4f} (peak {max(em):.4f})  "
        f"score {sc[0]:.4f} -> {sc[-1]:.4f}  wall_time_s={wall_time_s}"
    )


if __name__ == "__main__":
    main()

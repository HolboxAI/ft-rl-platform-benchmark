#!/usr/bin/env python3
"""Extract the per-step CholecT50 RL metrics from a Miles run log.

Parses the ``metrics.py:56 - eval N: {...}`` lines (greedy val reward) and the
``metrics.py:94 - perf N: {...}`` lines (rollout reward + component F1s), then
writes ``miles/metrics/cholect50_rl_final_results.json`` in the canonical
per-experiment shape.

The reported number is the val greedy mean@1 (``eval/cholect50_val``), which is
the F1-blended reward the verl reference reports (target 0.715). The rollout
curve (``rollout/raw_reward_unfiltered``) is the GRPO training signal over
``n_samples_per_prompt=8`` sampled completions, so it is noisier than the val.

Usage:  python3 extract_cholect50_rl_metrics.py <run_cholect50.log>
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(os.path.dirname(HERE), "metrics")

ANSI = re.compile(r"\x1b\[[0-9;]*m")

EVAL_RE = re.compile(r"eval \d+:\s*(\{.*)")
PERF_RE = re.compile(r"perf \d+:\s*(\{.*)")
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _parse(line: str, pat) -> dict | None:
    m = pat.search(line)
    if not m:
        return None
    raw = ANSI.sub("", m.group(1)).strip()
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None


def parse_log(path: str):
    val_reward, rollout_reward = [], []
    triplet_f1, inst_f1, verb_f1, targ_f1 = [], [], [], []
    timestamps = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ev = _parse(line, EVAL_RE)
            if ev is not None and "eval/cholect50_val" in ev:
                val_reward.append(ev["eval/cholect50_val"])
                ts = TS_RE.search(line)
                if ts:
                    timestamps.append(ts.group(1))
                continue

            pf = _parse(line, PERF_RE)
            if pf is not None and "rollout/raw_reward_unfiltered" in pf:
                rollout_reward.append(pf["rollout/raw_reward_unfiltered"])
                triplet_f1.append(pf["rollout/triplet_f1"])
                inst_f1.append(pf["rollout/instrument_f1"])
                verb_f1.append(pf["rollout/verb_f1"])
                targ_f1.append(pf["rollout/target_f1"])
                ts = TS_RE.search(line)
                if ts:
                    timestamps.append(ts.group(1))

    return {
        "val_reward_curve": val_reward,
        "rollout_reward_curve": rollout_reward,
        "triplet_f1_curve": triplet_f1,
        "instrument_f1_curve": inst_f1,
        "verb_f1_curve": verb_f1,
        "target_f1_curve": targ_f1,
        "timestamps": timestamps,
    }


def _last(xs):
    return [x for x in xs if x is not None]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    d = parse_log(sys.argv[1])

    val = _last(d["val_reward_curve"])
    rr = _last(d["rollout_reward_curve"])
    t3 = _last(d["triplet_f1_curve"])
    ins = _last(d["instrument_f1_curve"])
    vb = _last(d["verb_f1_curve"])
    tg = _last(d["target_f1_curve"])
    ts = d["timestamps"]

    if not val or not rr:
        print(f"ERROR: no eval/perf metric lines found in {sys.argv[1]}")
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
        "experiment": "cholect50_rl",
        "method": "grpo",
        "platform": "miles",
        "framework": "miles (slime fork) — SGLang rollout + FSDP train",
        "job_id": "miles-spot-p4d24xlarge-cholect50-rl",
        "state": "COMPLETED",
        "base_model": "cholec-sft-qwen3.5-4b (Qwen3.5-4B VLM, model_type qwen3_5; Tinker SFT output, LoRA merged; same checkpoint as verl and SkyRL)",
        "hf_model_id": None,
        "lora_rank": None,
        "learning_rate": 1e-05,
        "kl_beta": 0.04,
        "kl_loss_type": "low_var_kl",
        "temperature": 0.7,
        "top_p": 1.0,
        "n_samples_per_prompt": 8,
        "rollout_batch_size": 16,
        "global_batch_size": 128,
        "max_output_tokens": 256,
        "max_prompt_tokens": 2048,
        "dataset_size": 256,
        "val_dataset_size": 32,
        "num_rollout_steps": len(rr),
        "reward": "miles/scripts/reward_cholect50.py (0.4*triplet_f1 + 0.6*mean(inst,verb,targ)_f1)",
        "start_val_reward": val[0],
        "final_val_reward": val[-1],
        "peak_val_reward": max(val),
        "start_rollout_reward": rr[0],
        "final_rollout_reward": rr[-1],
        "peak_rollout_reward": max(rr),
        "component_f1_final": {
            "triplet_f1": t3[-1] if t3 else None,
            "instrument_f1": ins[-1] if ins else None,
            "verb_f1": vb[-1] if vb else None,
            "target_f1": tg[-1] if tg else None,
        },
        "wall_time_s": wall_time_s,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "val_reward_curve": val,
        "rollout_reward_curve": rr,
        "triplet_f1_curve": t3,
        "instrument_f1_curve": ins,
        "verb_f1_curve": vb,
        "target_f1_curve": tg,
    }

    out = os.path.join(METRICS_DIR, "cholect50_rl_final_results.json")
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    print(
        f"steps={len(rr)}  val {val[0]:.4f} -> {val[-1]:.4f}  "
        f"rollout {rr[0]:.4f} -> {rr[-1]:.4f} (peak {max(rr):.4f})  "
        f"wall_time_s={wall_time_s}"
    )


if __name__ == "__main__":
    main()

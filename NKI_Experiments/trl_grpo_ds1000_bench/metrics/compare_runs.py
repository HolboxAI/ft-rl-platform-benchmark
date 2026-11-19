"""Compare runs across platforms (g6e vs trn1 / trn2 / trn3, with and without NKI).

Reads every results/<run>/run_summary.csv + phase_summary.csv (and any kernel_microbench.csv),
groups repeated runs of the same variant (platform + nki_ops + sync_mode; median across
repeats) and writes, into results/comparison/:

  all_runs.csv            every run_summary row, one per run
  variant_summary.csv     one row per variant (median of repeats)
  phase_comparison.csv    per phase x variant: median s/step, share of step, speedup vs baseline
  metric_comparison.csv   headline metrics x variant: value, ratio vs baseline, better/worse
  kernel_comparison.csv   op-level timings x platform/impl with speedup vs baseline platform

Speedup > 1 means faster than the baseline (default: the first variant whose platform
starts with 'g6e'). These files are the input for the visualisation step.

  python -m metrics.compare_runs                       # all runs under results/
  python -m metrics.compare_runs --baseline g6e.xlarge results/run_a results/run_b
"""

import argparse
import csv
import glob
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from metrics.csv_store import PHASES, read_csv  # noqa: E402

# metric -> True if higher is better
HEADLINE = {
    "step_time_median_s": False, "first_step_s": False, "warmup_total_s": False,
    "train_wall_s": False, "model_load_s": False,
    "rollout_total_median_s": False, "generation_median_s": False, "reward_median_s": False,
    "logprob_nograd_median_s": False, "loss_compute_median_s": False,
    "policy_forward_median_s": False, "backward_median_s": False, "optimizer_step_median_s": False,
    "gen_tokens_per_s_median": True, "train_tokens_per_s_median": True,
    "samples_per_s_median": True, "end_to_end_gen_tokens_per_s": True,
    "device_util_mean_pct": True, "device_mem_max_mb": False, "device_power_mean_w": False,
    "compile_time_s_warmup": False, "compile_time_s_measured": False,
    "cost_usd_per_step": False, "cost_usd_per_1M_gen_tokens": False, "cost_usd_full_run": False,
    "reward_mean_all": True,
}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _write(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})


def variant_of(r):
    return f'{r.get("platform")}|nki-{r.get("nki_ops")}|sync-{r.get("sync_mode")}'


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", help="run dirs (default: every results/*/run_summary.csv)")
    p.add_argument("--results", default=os.path.join(ROOT, "results"))
    p.add_argument("--baseline", default="g6e", help="platform label prefix used as baseline")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    runs = a.runs or sorted(os.path.dirname(x) for x in glob.glob(os.path.join(a.results, "*", "run_summary.csv")))
    if not runs:
        sys.exit("no runs with run_summary.csv found")
    out = a.out or os.path.join(a.results, "comparison")
    os.makedirs(out, exist_ok=True)

    summaries, phases = [], []
    for d in runs:
        s = read_csv(os.path.join(d, "run_summary.csv"))[0]
        s["variant"] = variant_of(s)
        summaries.append(s)
        ph = os.path.join(d, "phase_summary.csv")
        if os.path.exists(ph):
            for r in read_csv(ph):
                r["variant"] = s["variant"]
                phases.append(r)
    _write(os.path.join(out, "all_runs.csv"), summaries)

    # ---- variant_summary: median over repeats
    variants = {}
    for s in summaries:
        variants.setdefault(s["variant"], []).append(s)
    vrows = []
    for v, rs in variants.items():
        row = {"variant": v, "platform": rs[0]["platform"], "backend": rs[0]["backend"],
               "nki_ops": rs[0]["nki_ops"], "sync_mode": rs[0]["sync_mode"], "repeats": len(rs)}
        for k in rs[0]:
            vals = [_f(r.get(k)) for r in rs]
            if any(x is not None for x in vals) and k not in row:
                row[k] = _median(vals)
        vrows.append(row)
    _write(os.path.join(out, "variant_summary.csv"), vrows)

    base = next((r for r in vrows if str(r["platform"]).startswith(a.baseline)), vrows[0])
    print(f"[compare] baseline variant: {base['variant']}")

    # ---- metric_comparison
    mrows = []
    for metric, higher_better in HEADLINE.items():
        b = _f(base.get(metric))
        for r in vrows:
            v = _f(r.get(metric))
            ratio = (v / b) if (v is not None and b) else None
            speed = None
            if ratio:
                speed = ratio if higher_better else (1 / ratio if ratio else None)
            mrows.append({"metric": metric, "higher_is_better": int(higher_better),
                          "variant": r["variant"], "platform": r["platform"], "nki_ops": r["nki_ops"],
                          "sync_mode": r["sync_mode"], "value": v, "baseline_value": b,
                          "ratio_vs_baseline": ratio, "advantage_vs_baseline": speed,
                          "better_than_baseline": "" if speed is None or r is base else
                          ("yes" if speed > 1.0 else "no")})
    _write(os.path.join(out, "metric_comparison.csv"), mrows)

    # ---- phase_comparison
    by = {}
    for r in phases:
        by.setdefault((r["variant"], r["phase"]), []).append(r)
    prows = []
    for ph in PHASES + ["other"]:
        bvals = by.get((base["variant"], ph), [])
        bmed = _median([_f(x["median_s_per_step"]) for x in bvals])
        for v in variants:
            rs = by.get((v, ph), [])
            if not rs:
                continue
            med = _median([_f(x["median_s_per_step"]) for x in rs])
            speedup = (bmed / med) if (bmed and med) else None
            prows.append({"phase": ph, "parent": rs[0].get("parent", ""), "variant": v,
                          "platform": rs[0]["platform"], "nki_ops": rs[0]["nki_ops"],
                          "sync_mode": rs[0]["sync_mode"],
                          "median_s_per_step": med,
                          "p90_s_per_step": _median([_f(x["p90_s_per_step"]) for x in rs]),
                          "first_call_s": _median([_f(x["first_call_s"]) for x in rs]),
                          "share_of_step_pct": _median([_f(x["share_of_step_pct"]) for x in rs]),
                          "baseline_median_s_per_step": bmed,
                          "speedup_vs_baseline": speedup,
                          "faster_than_baseline": "" if speedup is None or v == base["variant"]
                          else ("yes" if speedup > 1.0 else "no")})
    _write(os.path.join(out, "phase_comparison.csv"), prows)

    # ---- kernel_comparison (from kernels/kernel_microbench.py outputs)
    kfiles = glob.glob(os.path.join(a.results, "**", "kernel_microbench.csv"), recursive=True)
    krows = [r for f in kfiles for r in read_csv(f) if r.get("status") == "ok"]
    if krows:
        base_k = {}
        for r in krows:
            if r["platform"].startswith(a.baseline) and r["impl"] == "torch":
                base_k[(r["op"], r["shape"])] = _f(r["median_ms"])
        out_k = []
        for r in krows:
            b = base_k.get((r["op"], r["shape"]))
            m = _f(r["median_ms"])
            out_k.append({**r, "baseline_median_ms": b,
                          "speedup_vs_baseline": (b / m) if (b and m) else None})
        _write(os.path.join(out, "kernel_comparison.csv"), out_k)

    print(f"[compare] {len(runs)} runs, {len(vrows)} variants -> {out}")


if __name__ == "__main__":
    main()

"""Turn one run's raw CSVs into summary CSVs (no plotting; those come later).

Reads   run_info.csv, step_metrics.csv, phase_events.csv, rewards.csv, device_util.csv
Writes  phase_summary.csv       one row per phase: per-step mean/median/p90, first call,
                                share of step time (warm-up steps excluded)
        step_device_util.csv    accelerator utilisation / memory per step
        reward_by_library.csv   pass rate and execution time per DS-1000 library
        run_summary.csv         one row with the headline numbers for cross-platform comparison

  python -m metrics.summarize_run results/<run_dir>
"""

import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.csv_store import PHASES, read_csv, read_run_info  # noqa: E402

PARENT = {"generation": "rollout_total", "reward": "rollout_total",
          "logprob_nograd": "rollout_total", "policy_forward": "loss_compute"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pct(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "median": None, "p90": None, "min": None, "max": None, "std": None}
    return {"mean": statistics.fmean(vals), "median": statistics.median(vals), "p90": _pct(vals, 0.9),
            "min": min(vals), "max": max(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}


def _write(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else ("" if v is None else v))
                        for k, v in r.items()})


def _load(run_dir, name):
    p = os.path.join(run_dir, name)
    return read_csv(p) if os.path.exists(p) and os.path.getsize(p) > 0 else []


def summarize(run_dir):
    info = read_run_info(run_dir)
    steps = _load(run_dir, "step_metrics.csv")
    events = _load(run_dir, "phase_events.csv")
    rewards = _load(run_dir, "rewards.csv")
    dev = _load(run_dir, "device_util.csv")
    label = {"platform": info.get("platform_label"), "backend": info.get("backend"),
             "nki_ops": info.get("nki_ops"), "sync_mode": info.get("sync_mode"),
             "run_name": info.get("run_name")}

    measured = [s for s in steps if s.get("is_warmup") == "0"] or steps
    warm = [s for s in steps if s.get("is_warmup") == "1"]
    mean_step = statistics.fmean([_f(s["t_step_total_s"]) for s in measured]) if measured else None

    # ---------------- phase_summary.csv
    first_call = {}
    for e in events:
        if e["call_idx"] == "0":
            first_call[e["phase"]] = _f(e["duration_s"])
    calls = {}
    for e in events:
        calls[e["phase"]] = calls.get(e["phase"], 0) + 1
    phase_rows = []
    for ph in PHASES + ["other"]:
        col = f"t_{ph}_s"
        per_step = [_f(s.get(col)) for s in measured]
        st = _stats(per_step)
        phase_rows.append({**label, "phase": ph, "parent": PARENT.get(ph, "step_total" if ph != "step_total" else ""),
                           "calls_total": calls.get(ph, ""),
                           "mean_s_per_step": st["mean"], "median_s_per_step": st["median"],
                           "p90_s_per_step": st["p90"], "min_s_per_step": st["min"],
                           "max_s_per_step": st["max"], "std_s_per_step": st["std"],
                           "first_call_s": first_call.get(ph),
                           "warmup_mean_s_per_step": _stats([_f(s.get(col)) for s in warm])["mean"],
                           "share_of_step_pct": (100 * st["mean"] / mean_step) if (st["mean"] is not None and mean_step) else None})
    _write(os.path.join(run_dir, "phase_summary.csv"), phase_rows)

    # ---------------- step_device_util.csv
    step_util = []
    for s in steps:
        t0, t1 = _f(s["step_start_unix"]), _f(s["step_end_unix"])
        smp = [d for d in dev if t0 is not None and t0 <= _f(d["timestamp_unix"]) <= t1]
        u = [_f(d["util_pct"]) for d in smp if _f(d["util_pct"]) is not None]
        m = [_f(d["mem_used_mb"]) for d in smp if _f(d["mem_used_mb"]) is not None]
        pw = [_f(d["power_w"]) for d in smp if _f(d["power_w"]) is not None]
        step_util.append({**label, "step": s["step"], "is_warmup": s["is_warmup"], "samples": len(smp),
                          "util_mean_pct": statistics.fmean(u) if u else None,
                          "util_max_pct": max(u) if u else None,
                          "mem_max_mb": max(m) if m else None,
                          "power_mean_w": statistics.fmean(pw) if pw else None})
    _write(os.path.join(run_dir, "step_device_util.csv"), step_util)

    # ---------------- reward_by_library.csv
    libs = {}
    for r in rewards:
        libs.setdefault(r["library"], []).append(r)
    lib_rows = []
    for lib, rs in sorted(libs.items()):
        lib_rows.append({**label, "library": lib, "completions": len(rs),
                         "pass_rate": statistics.fmean(_f(r["passed"]) for r in rs),
                         "exec_time_mean_s": statistics.fmean(_f(r["exec_time_s"]) for r in rs),
                         "timeouts": sum(r["status"] == "Timeout" for r in rs)})
    _write(os.path.join(run_dir, "reward_by_library.csv"), lib_rows)

    # ---------------- run_summary.csv
    def med(col, rows=measured):
        return _stats([_f(s.get(col)) for s in rows])["median"]

    def total(col, rows):
        v = [_f(s.get(col)) for s in rows if _f(s.get(col)) is not None]
        return sum(v) if v else None

    gen_tok = total("generated_tokens", measured)
    meas_time = total("t_step_total_s", measured)
    price = _f(info.get("instance_price_usd_hr"))
    meas_util = [u for u in step_util if u["is_warmup"] == "0" and u["util_mean_pct"] is not None]
    summary = {
        **label,
        "instance_type": info.get("instance_type"), "device_name": info.get("device_name", "")[:80],
        "model": info.get("cfg_model_name"), "status": info.get("status"),
        "steps_total": len(steps), "steps_measured": len(measured),
        "model_load_s": _f(info.get("model_load_s")), "trainer_init_s": _f(info.get("trainer_init_s")),
        "train_wall_s": _f(info.get("train_wall_s")),
        "first_step_s": _f(steps[0]["t_step_total_s"]) if steps else None,
        "warmup_total_s": total("t_step_total_s", warm),
        "step_time_mean_s": mean_step, "step_time_median_s": med("t_step_total_s"),
        "step_time_p90_s": _stats([_f(s["t_step_total_s"]) for s in measured])["p90"],
        "step_time_std_s": _stats([_f(s["t_step_total_s"]) for s in measured])["std"],
    }
    for ph in PHASES[1:] + ["other"]:
        summary[f"{ph}_median_s"] = med(f"t_{ph}_s")
    summary.update({
        "gen_tokens_per_s_median": med("gen_tokens_per_s"),
        "train_tokens_per_s_median": med("train_tokens_per_s"),
        "samples_per_s_median": med("samples_per_s"),
        "end_to_end_gen_tokens_per_s": (gen_tok / meas_time) if gen_tok and meas_time else None,
        "reward_mean_all": statistics.fmean(_f(r["reward"]) for r in rewards) if rewards else None,
        "pass_rate_first_step": _f(steps[0].get("pass_rate")) if steps else None,
        "pass_rate_last_step": _f(steps[-1].get("pass_rate")) if steps else None,
        "frac_groups_zero_std_mean": _stats([_f(s.get("frac_groups_zero_std")) for s in steps])["mean"],
        "peak_mem_mb_max": _stats([_f(s.get("peak_mem_mb")) for s in steps])["max"],
        "device_mem_max_mb": max((u["mem_max_mb"] for u in step_util if u["mem_max_mb"]), default=None),
        "device_util_mean_pct": statistics.fmean(u["util_mean_pct"] for u in meas_util) if meas_util else None,
        "device_power_mean_w": _stats([u["power_mean_w"] for u in step_util if u["is_warmup"] == "0"])["mean"],
        "compile_count_warmup": total("compile_count_delta", warm),
        "compile_time_s_warmup": total("compile_time_s_delta", warm),
        "compile_count_measured": total("compile_count_delta", measured),
        "compile_time_s_measured": total("compile_time_s_delta", measured),
        "nki_kernel_calls_total": total("nki_kernel_calls", steps),
        "instance_price_usd_hr": price,
        "cost_usd_per_step": (price / 3600 * mean_step) if price and mean_step else None,
        "cost_usd_per_1M_gen_tokens": (price / 3600 * meas_time / gen_tok * 1e6) if price and gen_tok and meas_time else None,
        "cost_usd_full_run": (price / 3600 * _f(info.get("train_wall_s"))) if price and _f(info.get("train_wall_s")) else None,
    })
    _write(os.path.join(run_dir, "run_summary.csv"), [summary])
    print(f"[summarize] wrote phase_summary / step_device_util / reward_by_library / run_summary in {run_dir}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m metrics.summarize_run results/<run_dir>")
    summarize(sys.argv[1])

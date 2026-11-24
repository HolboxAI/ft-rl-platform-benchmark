"""Run every CPU experiment for attention_cte_fp32_scale_in_exp.patch and write results/*.json.

  python run_experiments.py            # all (about 1-2 min)
  python run_experiments.py --quick    # fewer seeds, for a smoke test

Experiments
  exp1  path selection per config, before vs after (kernel decision code)
  exp2  accuracy: scenario x head dim x input family x seed, before vs after
  exp3  accuracy vs outlier gain (d=128, prefix-caching scenario)
  exp4  MM1 post-processing instruction model on the causal diagonal
  exp5  multi-section flash + attention sink + out_neg_max, patch vs upstream (bit comparison)
"""
import json
import math
import os
import sys

import numpy as np

from common import (BF16, K_TILE, Q_TILE, TOL, Cfg, bf16, decide, kernel, make_inputs,
                    make_mask, reference, rel_err)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
QUICK = "--quick" in sys.argv
SEEDS = list(range(2 if QUICK else 5))
S = 1024 if QUICK else 2048


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", name)


# ------------------------------------------------------------------ exp1
def exp1():
    rows = []
    scen = {
        "plain causal": dict(),
        "prefix caching": dict(prefix_caching=True),
        "sliding window": dict(sliding_window=1024),
        "context parallel": dict(cp=True),
        "sequence packing": dict(seq_packed=True),
        "plain causal + position_bias": dict(position_bias=True),
        "prefix caching + position_bias": dict(prefix_caching=True, position_bias=True),
    }
    for d in (64, 80, 96, 128, 192, 256):
        for sname, kw in scen.items():
            c = Cfg(d=d, scale=1 / math.sqrt(d), **kw)
            b, a = decide(c, False), decide(c, True)
            rows.append({"d": d, "scale": c.scale, "pow2": math.frexp(c.scale)[0] == 0.5,
                         "scenario": sname, "before": b, "after": a})
    dump("exp1_path_selection", rows)


# ------------------------------------------------------------------ exp2
def run_modes(q, k, v, mask, scale, c):
    """Return (before_err, after_err, before_mode, after_mode) via the decision logic."""
    ref = reference(q, k, v, mask, scale)
    res = {}
    for tag, patched in (("before", False), ("after", True)):
        dec = decide(c, patched)
        if not dec["runs"]:
            res[tag] = None
            continue
        mode = {"MM1 evict, fp32": "evict", "Q, bf16, exact": "fold",
                "Q, bf16, rounded": "fold", "exp, fp32": "patch"}[dec["where"]]
        res[tag] = (mode, rel_err(kernel(q, k, v, mask, scale, mode), ref))
    return res


def exp2():
    scen = {  # name -> (Cfg kwargs, sq, prior, window)
        "plain causal": (dict(), S, 0, 0),
        "prefix caching": (dict(prefix_caching=True), S // 2, S // 2, 0),
        "sliding window": (dict(sliding_window=S // 2), S, 0, S // 2),
    }
    rows = []
    for d in (64, 80, 96, 128, 256):
        scale = 1 / math.sqrt(d)
        for fam in ("repo_uniform", "gaussian", "outlier"):
            for sname, (kw, sq, prior, win) in scen.items():
                c = Cfg(d=d, scale=scale, **kw)
                errs = {"before": [], "after": []}
                modes = {}
                for seed in SEEDS:
                    q, k, v = make_inputs(sq, S, d, fam, seed)
                    mask = make_mask(sq, S, prior, win)
                    r = run_modes(q, k, v, mask, scale, c)
                    for tag in ("before", "after"):
                        modes[tag] = r[tag][0]
                        errs[tag].append(r[tag][1])
                rows.append({"d": d, "family": fam, "scenario": sname, "modes": modes,
                             **{f"{t}_errs": errs[t] for t in errs},
                             **{f"{t}_worst": max(errs[t]) for t in errs},
                             **{f"{t}_pass": sum(e <= TOL for e in errs[t]) for t in errs},
                             "n_seeds": len(SEEDS)})
        print("  d =", d)
    dump("exp2_accuracy", {"S": S, "tol": TOL, "seeds": SEEDS, "rows": rows})


# ------------------------------------------------------------------ exp3
def exp3():
    d, scale = 128, 1 / math.sqrt(128)
    c = Cfg(d=d, scale=scale, prefix_caching=True)
    gains = [1, 2, 4, 6, 8, 12, 16]
    out = {"gains": gains, "before": [], "after": [], "evict_ref": []}
    for g in gains:
        eb, ea, ee = [], [], []
        for seed in SEEDS:
            q, k, v = make_inputs(S // 2, S, d, "outlier", seed, gain=float(g))
            mask = make_mask(S // 2, S, S // 2)
            ref = reference(q, k, v, mask, scale)
            eb.append(rel_err(kernel(q, k, v, mask, scale, "fold"), ref))
            ea.append(rel_err(kernel(q, k, v, mask, scale, "patch"), ref))
            ee.append(rel_err(kernel(q, k, v, mask, scale, "evict"), ref))
        out["before"].append(max(eb))
        out["after"].append(max(ea))
        out["evict_ref"].append(max(ee))
        print("  gain", g)
    dump("exp3_outlier_sweep", {"S": S, "tol": TOL, **out})


# ------------------------------------------------------------------ exp4
def exp4():
    rows = []
    for L in (512, 1024, 2048, 4096, 8192, 32768):
        nq, nk = L // Q_TILE, L // K_TILE
        full = diag = 0
        for i in range(nq):
            q0, q1 = i * Q_TILE, i * Q_TILE + Q_TILE - 1
            for j in range(nk):
                k0, k1 = j * K_TILE, j * K_TILE + K_TILE - 1
                if k0 > q1:
                    continue                      # entirely above the diagonal: skipped
                if k1 <= q0:
                    full += 1
                else:
                    diag += 1
        before, after = full + 3 * diag, full + diag
        rows.append({"L": L, "full": full, "diag": diag, "diag_share": diag / (full + diag),
                     "before": before, "after": after, "reduction": 1 - after / before,
                     "affine_elems_removed": diag * Q_TILE * K_TILE})
    dump("exp4_tile_model", rows)


# ------------------------------------------------------------------ exp5
def flash(q, k, v, scale, sections, sink, patched):
    """Section loop with running max/sum/correction, optional sink, and out_neg_max."""
    f32 = np.float32
    sc = f32(scale)
    sq, skv = q.shape[0], k.shape[0]
    sec = skv // sections
    m = np.full((sq, 1), f32(-3.0e38))               # running max of the scores held in SBUF
    l = np.zeros((sq, 1), f32)
    o = np.zeros((sq, v.shape[1]), f32)
    mask = make_mask(sq, skv, 0, 0)
    sink_v = None if sink is None else (f32(sink) / sc if patched else f32(sink))
    for si in range(sections):
        sl = slice(si * sec, (si + 1) * sec)
        s = (q @ k[sl].T).astype(f32)
        if not patched:
            s = (s * sc).astype(f32)
        s = np.where(mask[:, sl], s, f32(-3.0e38))
        m_new = np.maximum(m, s.max(1, keepdims=True))
        if si == sections - 1 and sink_v is not None:
            m_new = np.maximum(m_new, sink_v)
        if patched:
            corr = np.exp((sc * m + sc * (-m_new)).astype(f32)) if si else f32(0)
            e = np.exp((s * sc + (sc * (-m_new)).astype(f32)).astype(f32))
        else:
            corr = np.exp((m - m_new).astype(f32)) if si else f32(0)
            e = np.exp((s + (-m_new)).astype(f32))
        e = np.where(mask[:, sl], e, f32(0))
        l = (l * corr + e.sum(1, keepdims=True, dtype=f32)).astype(f32)
        o = (o * corr + (bf16(e) @ v[sl])).astype(f32)
        m = m_new
    if sink_v is not None:
        if patched:
            es = np.exp((sc * sink_v + (sc * (-m)).astype(f32)).astype(f32))
        else:
            es = np.exp((sink_v + (-m)).astype(f32))
        l = (l + es).astype(f32)
    neg_max = (sc * (-m)).astype(f32) if patched else (-m).astype(f32)
    # upstream stores -max of already-scaled scores; patch stores scale * (-max of unscaled)
    return o / l, neg_max


def exp5():
    d, scale = 128, 1 / math.sqrt(128)
    rows = []
    sq, skv = (256, 1024) if QUICK else (512, 2048)
    q, k, v = make_inputs(sq, skv, d, "outlier", 0)
    for sections in (1, 4, 8):
        for sink in (None, 2.5, 20.0):
            ou, nu = flash(q, k, v, scale, sections, sink, False)
            op, np_ = flash(q, k, v, scale, sections, sink, True)
            rows.append({"sections": sections, "sink": sink,
                         "max_out_diff": float(np.max(np.abs(ou - op))),
                         "max_negmax_diff": float(np.max(np.abs(nu - np_))),
                         "bit_identical": bool(np.array_equal(ou, op) and np.array_equal(nu, np_))})
    dump("exp5_sections_sink", {"sq": sq, "skv": skv, "rows": rows})


if __name__ == "__main__":
    for name, fn in (("exp1", exp1), ("exp2", exp2), ("exp3", exp3), ("exp4", exp4), ("exp5", exp5)):
        print(name)
        fn()

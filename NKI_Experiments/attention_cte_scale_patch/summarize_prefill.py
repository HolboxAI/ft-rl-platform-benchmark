"""Summarise results_device/prefill_latency/ntff/<scale>_<tree>_r<rep>/s<len>.json -> summary.json + table."""
import glob, json, os, re, statistics as st
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_device", "prefill_latency")
KEYS = ["total_time", "neuroncore_cycle_count", "tensor_engine_instruction_count", "vector_engine_instruction_count",
        "scalar_engine_instruction_count", "gpsimd_engine_instruction_count"]
data = {}
for f in glob.glob(os.path.join(B, "ntff", "*", "s*.json")):
    tag, tree, rep = re.match(r"(real|ctrl)_(before|after)_r(\d+)", os.path.basename(os.path.dirname(f))).groups()
    p = json.load(open(f))["device_profile_profile"]
    data.setdefault((tag, int(os.path.basename(f)[1:-5]), tree), {})[int(rep)] = {k: p.get(k) for k in KEYS}
out = []
for tag in ("real", "ctrl"):
    print(f"\n== {tag}: ms median [min-max]; delta = after vs before")
    for s in sorted({k[1] for k in data if k[0] == tag}):
        b, a = data[(tag, s, "before")], data[(tag, s, "after")]
        tb = [b[r]["total_time"] * 1e3 for r in sorted(b)]; ta = [a[r]["total_time"] * 1e3 for r in sorted(a)]
        paired = [(a[r]["total_time"] / b[r]["total_time"] - 1) * 100 for r in sorted(b) if r in a]
        row = dict(scale=tag, seqlen=s, n=len(paired), before_ms=st.median(tb), after_ms=st.median(ta),
                   before_range=[min(tb), max(tb)], after_range=[min(ta), max(ta)],
                   delta_pct=(st.median(ta) / st.median(tb) - 1) * 100, paired_delta_pct=paired,
                   counts={k: [b[1][k], a[1][k]] for k in KEYS[2:]})
        out.append(row)
        print(f"{s:5d} n={len(paired)} before {row['before_ms']:.4f} [{min(tb):.4f}-{max(tb):.4f}]  after {row['after_ms']:.4f} [{min(ta):.4f}-{max(ta):.4f}]  "
              f"delta {row['delta_pct']:+.2f}%  paired {min(paired):+.2f}..{max(paired):+.2f}")
        print("      instr before/after  " + "  ".join(f"{k.split('_')[0]} {v[0]}/{v[1]}" for k, v in row["counts"].items()))
json.dump(out, open(os.path.join(B, "summary.json"), "w"), indent=1)

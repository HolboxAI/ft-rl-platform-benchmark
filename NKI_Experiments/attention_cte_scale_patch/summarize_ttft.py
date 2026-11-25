"""Summarise results_device/e2e_ttft/<before|after>_<len>_r<rep>.json -> summary.json + table."""
import glob, json, os, re, statistics as st
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_device", "e2e_ttft")
d = {}
for f in glob.glob(os.path.join(B, "*_r*.json")):
    m = re.match(r"(before|after)_(\d+)_r(\d+)\.json$", os.path.basename(f))
    if m:
        r = json.load(open(f))
        assert os.sep + "work" in r["kernel_file"].replace("/", os.sep) and f"{os.sep}{m[1]}{os.sep}" in r["kernel_file"].replace("/", os.sep)
        d.setdefault((int(m[2]), m[1]), {})[int(m[3])] = st.median(r["ttft_ms"])
out = []
for S in sorted({k[0] for k in d}):
    b, a = d.get((S, "before"), {}), d.get((S, "after"), {})
    reps = sorted(set(a) & set(b))
    pairs = [(a[i] / b[i] - 1) * 100 for i in reps]
    mb, ma = st.median(b[i] for i in reps), st.median(a[i] for i in reps)
    ref = json.load(open(os.path.join(B, f"ref_{S}.json")))
    out.append(dict(prompt_len=S, pairs=len(reps), before_ttft_ms=mb, after_ttft_ms=ma, delta_pct=(ma / mb - 1) * 100,
                    before_range=[min(b[i] for i in reps), max(b[i] for i in reps)],
                    after_range=[min(a[i] for i in reps), max(a[i] for i in reps)], paired_delta_pct=pairs,
                    torch_reference_ttft_ms=st.median(ref["ttft_ms"])))
    print(f"{S:5d} pairs={len(reps)} before {mb:.2f} {[round(x,2) for x in out[-1]['before_range']]} after {ma:.2f} "
          f"{[round(x,2) for x in out[-1]['after_range']]} delta {out[-1]['delta_pct']:+.2f}% paired {min(pairs):+.2f}..{max(pairs):+.2f}")
json.dump(out, open(os.path.join(B, "summary.json"), "w"), indent=1)

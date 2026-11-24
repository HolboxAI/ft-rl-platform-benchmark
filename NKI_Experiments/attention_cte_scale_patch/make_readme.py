"""Write README.md (the report, in Markdown) from results/*.json and results_device/.
Run make_report.py first so images/*.png exist. Stdlib only.
"""
import json
import math
import os
import re
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
D = os.path.join(HERE, "results_device")


def load(n):
    with open(os.path.join(R, n + ".json")) as f:
        return json.load(f)


def md_table(head, rows):
    out = ["| " + " | ".join(str(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def device_cases(label):
    root = ET.parse(os.path.join(D, label + ".xml")).getroot()
    ts = root if root.tag == "testsuite" else root[0]
    res = {}
    for tc in ts.iter("testcase"):
        bad = tc.find("failure") is not None or tc.find("error") is not None
        res[tc.get("name")] = "FAIL" if bad else ("skipped" if tc.find("skipped") is not None else "pass")
    return res


def describe(name):
    short = name.split("[")[0].replace("test_attention_cte_", "").replace("test_", "")
    key = name.split("[", 1)[1] if "[" in name else ""
    s = re.search(r"_(?:skv|s)-(\d+)", key)
    d = re.search(r"_d-(\d+)", key)
    sink = ", sink" if "sink-1" in key else ""
    if not s:
        return f"`{short}`"
    return f"`{short}`, S={s.group(1)}, d={d.group(1) if d else '?'}{sink}"


def main():
    e1, e2, e3, e4, e5 = (load(n) for n in ("exp1_path_selection", "exp2_accuracy", "exp3_outlier_sweep",
                                           "exp4_tile_model", "exp5_sections_sink"))
    rows2, tol = e2["rows"], e2["tol"]
    pc = next(r for r in rows2 if r["d"] == 128 and r["family"] == "outlier" and r["scenario"] == "prefix caching")
    identical = [r for r in rows2 if r["d"] in (64, 256) and r["before_worst"] == r["after_worst"]]
    n_pow2 = len([r for r in rows2 if r["d"] in (64, 256)])
    b, a = device_cases("before"), device_cases("after")
    nb = sum(v == "pass" for v in b.values())
    na = sum(v == "pass" for v in a.values())
    nskip = sum(v == "skipped" for v in a.values())
    cyc = json.load(open(os.path.join(D, "cycles_apc.json")))
    cyc_line = ""
    for k, v in cyc.items():
        cyc_line = (f"Prefix caching with sink, S=2048: **{v['before'][0]:,.0f}** cycles before, "
                    f"**{v['after'][0]:,.0f}** after. An earlier run of the same case gave 84.66 M vs 86.46 M "
                    f"(+2.1%), so differences of that size are run-to-run noise, not the patch. "
                    f"(The cycle threshold on the new prefix-caching test rows was set to 1e12 in both trees to make the harness record cycles; that edit was made on the instance and is not part of the patch.)")

    # path-selection table: d=128 only
    p_rows = []
    for r in e1:
        if r["d"] != 128:
            continue
        def cell(x):
            if not x["runs"]:
                return "**fails**: " + x["note"]
            return x["where"] + (", fast diagonal" if x["fast_diag"] else "")
        p_rows.append([r["scenario"], cell(r["before"]), cell(r["after"])])

    acc_rows = []
    for r in rows2:
        if r["d"] in (64, 128, 256) and r["family"] == "outlier":
            acc_rows.append([r["d"], r["scenario"], f"{r['before_worst']:.2e}", f"{r['before_pass']}/{r['n_seeds']}",
                             f"{r['after_worst']:.2e}", f"{r['after_pass']}/{r['n_seeds']}"])

    iv = [[x["L"], x["full"], x["diag"], f"{x['diag_share']:.0%}", x["before"], x["after"], f"-{x['reduction']:.0%}"]
          for x in e4]
    sec = [[x["sections"], "none" if x["sink"] is None else x["sink"], x["max_out_diff"], x["max_negmax_diff"],
            "yes" if x["bit_identical"] else "NO"] for x in e5["rows"]]
    dev = [[describe(n), b[n], a.get(n, "skipped")] for n in b]

    md = f"""# attention_cte: non-power-of-two softmax scale, before vs after the patch

Validation of `attention_cte_fp32_scale_in_exp.patch` (base `ad2f1b9` of
[aws-neuron/nki-library](https://github.com/aws-neuron/nki-library), kernel
`src/nkilib_src/nkilib/core/attention/attention_cte.py`).

**Sections 1-5:** CPU emulation (numpy, bf16) of the kernel's arithmetic and decision logic.
**Section 6:** measured on Trainium2 hardware (trn2.48xlarge), running the patch's new test cases against both
the original and the patched kernel.

## Summary

| Result | Before | After | Source |
|---|---|---|---|
| Sequence packing, d=128, scale 1/sqrt(128) | **fails** (`range_select path doesn't support scale != 1.0`) | passes | trn2 hardware |
| New test cases | {nb} pass, {len(b)-nb-nskip} fail, {nskip} skipped | {na} pass, {len(a)-na-nskip} fail, {nskip} skipped | trn2 hardware |
| Prefix caching, d=128, outlier inputs: seeds within tolerance ({tol:g}) | {pc['before_pass']}/{pc['n_seeds']} | {pc['after_pass']}/{pc['n_seeds']} | CPU emulation |
| Power-of-two scales (d=64, 256) | exact | identical ({len(identical)} of {n_pow2} cases) | CPU emulation |
| Diagonal MM1 instructions, 512 to 4K tokens | 3 per diagonal tile | 1 per tile: -{e4[3]['reduction']:.0%} to -{e4[0]['reduction']:.0%} | instruction-count model, not a timing |

## The problem

Attention computes `softmax(scale * Q.K^T)`. The `range_select` masking instruction has no scale operand, so on the
prefix-caching, sliding-window, context-parallel and `kv_used_len` paths the kernel folded the scale into the bf16 Q
tile. For a power-of-two scale that is exact. For the scale most LLMs use at head size 128, 1/sqrt(128) = 0.0884, it
rounds every Q element. Sequence packing also uses `range_select` but was never added to the fold, so any scale other
than 1.0 hit `kernel_assert(ac.scale == 1.0)`. Plain causal prefill with a non-power-of-two scale could not use the
fast one-instruction diagonal.

## The patch

For a positive scale, keep the MM1 scores unscaled in fp32 and apply the scale inside the exp activation:
`exp(scale*s - scale*max)`. The flash-attention correction factor, the attention sink and the cached `out_neg_max`
are updated to match. Power-of-two scales keep the exact Q fold, and configs that do not need the new path emit the
same instructions as before.

## 1. Which path each config takes (d = 128, scale = 1/sqrt(128))

{md_table(["Scenario", "Before", "After"], p_rows)}

## 2. Accuracy vs outlier strength (CPU emulation)

Worst of {len(e2['seeds'])} seeds, S = {e3['S']}, prefix caching, d = 128. Error = max|out - ref| / max|ref| against
float64 attention on the same bf16 inputs; the repo tolerance is {tol:g}. Outlier inputs (4 of 128 q/k channels
scaled) are a stylised assumption, not measured LLM activations.

![Error vs outlier gain](images/chart1_error_vs_outlier_gain.png)

{md_table(["Outlier gain"] + e3["gains"], [["Before (fold into bf16 Q)"] + [f"{x:.1e}" for x in e3["before"]],
                                              ["After (scale in fp32 exp)"] + [f"{x:.1e}" for x in e3["after"]]])}

## 3. Accuracy by head dimension (CPU emulation, outlier inputs)

![Error by head dimension](images/chart2_error_by_head_dim.png)

{md_table(["d", "scenario", "Before: worst err", "pass", "After: worst err", "pass"], acc_rows)}

The repo's own tests use U[0,1) inputs, where attention is nearly uniform and every variant passes, so they cannot
show this difference.

## 4. Instruction count on the causal diagonal (model, not a timing)

128-row Q groups, 512-key tiles, tiles above the diagonal skipped. A diagonal tile goes from 3 instructions
(`tensor_copy` + GpSimd `affine_select` + evict) to 1 (`range_select`).

![Instruction model](images/chart3_instruction_model.png)

{md_table(["Prompt", "Full tiles", "Diagonal tiles", "Diagonal share", "Before", "After", "Reduction"], iv)}

## 5. Flash sections, sink and `out_neg_max` (CPU emulation)

Patch-style and upstream-style section loops compared bit for bit (d = 128). This shows the algebra is consistent
in emulation; on hardware a fused multiply-add could differ in the last fp32 bit.

{md_table(["Sections", "Sink", "max abs out diff", "max abs neg_max diff", "Bit-identical"], sec)}

## 6. Device run: trn2.48xlarge, same test cases against both kernels

NKI 0.7.0, neuronx-cc 2.28, compile-and-infer on Trainium2. "Before" = original kernel with the patch's new test
file; "after" = patched kernel. Raw logs: [`results_device/`](results_device/).

{md_table(["Case", "Before", "After"], dev)}

{cyc_line}

**Not measured on device:**
- The plain-causal diagonal speedup. The test harness records cycles only for the prefix-caching group (the
  plain-causal test has a `FIXME` for cycles), so the section 4 numbers are still a model.
- Accuracy with outlier inputs. The suite's inputs are U[0,1), where every variant passes.
- The sweep case `test_ranged_attn_cte_sweep_nocp` was skipped in both runs.
- TTFT and tokens per second. The patch only touches prefill attention, and the GRPO benchmark in
  `../trl_grpo_ds1000_bench` does not call this kernel, so those benchmark numbers are unaffected today.

## Reproduce

```bash
pip install numpy ml_dtypes matplotlib
python run_experiments.py        # CPU experiments (about 1 minute; --quick for a smoke test)
python make_report.py            # charts in images/ (also writes an HTML report, not committed)
python make_readme.py            # regenerate this README
bash device/run_before_after.sh <path to nki-library-main>   # on a trn2 instance
```

| File | Purpose |
|---|---|
| `attention_cte_fp32_scale_in_exp.patch` | the patch (2 files, +100 / -22) |
| `common.py` | bf16 kernel emulation; scale decision logic before and after the patch |
| `run_experiments.py` | the five CPU experiments -> `results/*.json` |
| `make_report.py`, `make_readme.py` | charts and this README |
| `device/run_before_after.sh` | trn2 runbook (needs the venv `bin` on `PATH`, otherwise every compile fails with `python None`) |
| `results/`, `results_device/`, `images/` | CPU results, device logs and junit XML, chart PNGs |
"""
    with open(os.path.join(HERE, "README.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("wrote README.md")


if __name__ == "__main__":
    main()

"""Build report.html (+ images/*.png) from results/*.json. Run run_experiments.py first."""
import base64
import html
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, IMG = os.path.join(HERE, "results"), os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)
BEFORE, AFTER, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
esc = html.escape


def load(n):
    with open(os.path.join(RES, n + ".json")) as f:
        return json.load(f)


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=140, bbox_inches="tight", facecolor="white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img(b64, alt):
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{b64}">'


# ------------------------------------------------------------------ charts
def chart_outliers(e3):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    style(ax)
    g = e3["gains"]
    ax.plot(g, e3["before"], "-o", color=BEFORE, lw=2, ms=6, label="Before: scale folded into bf16 Q")
    ax.plot(g, e3["after"], "-s", color=AFTER, lw=2, ms=6, label="After: scale applied in fp32 exp")
    ax.axhline(e3["tol"], color=MUTED, ls="--", lw=1.2)
    ax.text(g[0], e3["tol"] * 1.12, "repo tolerance 2e-2", color=MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("q/k outlier-channel gain (4 of 128 channels)", color=MUTED)
    ax.set_ylabel("max error vs float64 reference", color=MUTED)
    ax.set_title("Prefix caching, d = 128, scale = 1/sqrt(128)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 0.88))
    return save(fig, "chart1_error_vs_outlier_gain.png")


def chart_dims(rows, tol):
    sel = [r for r in rows if r["scenario"] == "prefix caching" and r["family"] == "outlier"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax)
    xs = range(len(sel))
    w = 0.36
    ax.bar([x - w / 2 for x in xs], [r["before_worst"] for r in sel], w, color=BEFORE, label="Before")
    ax.bar([x + w / 2 for x in xs], [r["after_worst"] for r in sel], w, color=AFTER, label="After")
    ax.axhline(tol, color=MUTED, ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    labs = []
    for r in sel:
        labs.append(f"d={r['d']}\n{'pow2' if r['d'] in (64, 256) else 'non-pow2'}")
    ax.set_xticklabels(labs)
    for x, r in zip(xs, sel):
        ax.text(x - w / 2, r["before_worst"] * 1.1, f"{r['before_pass']}/{r['n_seeds']}", ha="center", fontsize=8, color=MUTED)
        ax.text(x + w / 2, r["after_worst"] * 1.1, f"{r['after_pass']}/{r['n_seeds']}", ha="center", fontsize=8, color=MUTED)
    ax.set_ylabel("worst error over seeds", color=MUTED)
    ax.set_title("Prefix caching, outlier inputs (labels: seeds passing tolerance)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return save(fig, "chart2_error_by_head_dim.png")


def chart_instr(rows):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    style(ax)
    xs = range(len(rows))
    w = 0.36
    ax.bar([x - w / 2 for x in xs], [r["before"] for r in rows], w, color=BEFORE, label="Before (3 instr / diagonal tile)")
    ax.bar([x + w / 2 for x in xs], [r["after"] for r in rows], w, color=AFTER, label="After (1 instr / diagonal tile)")
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{r['L'] // 1024}K" if r["L"] >= 1024 else str(r["L"]) for r in rows])
    for x, r in zip(xs, rows):
        ax.text(x + w / 2, r["after"] * 1.12, f"-{r['reduction']:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xlabel("prompt length (tokens), causal prefill, d = 128", color=MUTED)
    ax.set_ylabel("MM1 evict/mask instructions per head", color=MUTED)
    ax.set_title("Instruction-count model, not a timing", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return save(fig, "chart3_instruction_model.png")


# ------------------------------------------------------------------ tables
def table(head, rows, cls=""):
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><tr>{th}</tr>{body}</table></div>'


def badge(ok, text):
    return f'<span class="b {"ok" if ok else "bad"}">{"&#10003;" if ok else "&#10007;"} {esc(text)}</span>'


def paths_table(rows, d_list=(64, 128, 256)):
    out = []
    for r in rows:
        if r["d"] not in d_list:
            continue
        b, a = r["before"], r["after"]

        def cell(x):
            if not x["runs"]:
                return badge(False, "fails: " + x["note"])
            return esc(x["where"]) + (" &middot; fast diagonal" if x["fast_diag"] else "")

        changed = (b["where"], b["fast_diag"], b["runs"]) != (a["where"], a["fast_diag"], a["runs"])
        out.append([r["d"], f"{r['scale']:.4f}" + (" (pow2)" if r["pow2"] else ""), esc(r["scenario"]),
                    cell(b), cell(a), "changed" if changed else ""])
    return table(["d", "scale", "scenario", "Before", "After", ""], out, "paths")


def acc_table(rows, tol):
    out = []
    for r in rows:
        if r["d"] not in (64, 128, 256):
            continue
        out.append([r["d"], esc(r["family"]), esc(r["scenario"]),
                    f"{r['before_worst']:.2e}", badge(r["before_pass"] == r["n_seeds"], f"{r['before_pass']}/{r['n_seeds']}"),
                    f"{r['after_worst']:.2e}", badge(r["after_pass"] == r["n_seeds"], f"{r['after_pass']}/{r['n_seeds']}")])
    return table(["d", "inputs", "scenario", "Before: worst err", "pass", "After: worst err", "pass"], out)


def device_section():
    """Device run (trn2.48xlarge): same new test cases against the original and the patched kernel."""
    import xml.etree.ElementTree as ET
    d = os.path.join(HERE, "results_device")
    if not os.path.exists(os.path.join(d, "before.xml")):
        return "<div class='card'><h2>6. Device run</h2><div class='sub'>Not run yet. See device/run_before_after.sh.</div></div>"

    def cases(label):
        root = ET.parse(os.path.join(d, label + ".xml")).getroot()
        ts = root if root.tag == "testsuite" else root[0]
        out = {}
        for tc in ts.iter("testcase"):
            st = "fail" if tc.find("failure") is not None or tc.find("error") is not None else (
                "skip" if tc.find("skipped") is not None else "pass")
            out[tc.get("name")] = st
        return out

    b, a = cases("before"), cases("after")
    rows = []
    for name in b:
        short = name.split("[")[0].replace("test_attention_cte_", "").replace("test_", "")
        key = name.split("[", 1)[1][:-1] if "[" in name else ""
        import re
        sc = re.search(r"_(?:skv|s)-(\d+)", key)
        dd = re.search(r"_d-(\d+)", key)
        extra = "sink" if "sink-1" in key else ""
        desc = f"{short}, S={sc.group(1) if sc else '?'}, d={dd.group(1) if dd else '?'}{', ' + extra if extra else ''}"

        def cell(s):
            return {"pass": badge(True, "pass"), "fail": badge(False, "FAIL"),
                    "skip": "<span class='pend'>skipped</span>"}[s]

        rows.append([esc(desc), cell(b[name]), cell(a.get(name, "skip"))])
    nb = sum(v == "pass" for v in b.values())
    na = sum(v == "pass" for v in a.values())
    cyc = ""
    try:
        cj = json.load(open(os.path.join(d, "cycles_apc.json")))
        for k, v in cj.items():
            cyc += (f"<p>Cycles, prefix caching with sink, S=2048: before {v['before'][0]:,.0f}, after "
                    f"{v['after'][0]:,.0f}. An earlier run of the same case gave 84.66 M vs 86.46 M "
                    f"(+2.1%), so single-run differences of that size are noise, not the patch.</p>")
    except Exception:
        pass
    return f"""<div class="card"><h2>6. Device run: trn2.48xlarge, same test cases against both kernels</h2>
<div class="sub">NKI 0.7.0, neuronx-cc 2.28, compile-and-infer on Trainium2. "Before" = original kernel with the patch's new test file; "after" = patched kernel. {nb} passed before, {na} after.</div>
{table(["Case", "Before", "After"], rows)}
{cyc}
<p class="sub"><b>Not measured:</b> the plain-causal diagonal speedup. The test harness only records cycles for the prefix-caching group (the plain-causal test has a FIXME for cycles), so the instruction-count model in section 4 is still a model. Accuracy with outlier inputs on the device is not measured either: the suite's inputs are U[0,1), where every variant passes. The sweep case (<code>test_ranged_attn_cte_sweep_nocp</code>) was skipped.</p></div>"""


def main():
    e1, e2, e3, e4, e5 = (load(n) for n in ("exp1_path_selection", "exp2_accuracy", "exp3_outlier_sweep",
                                           "exp4_tile_model", "exp5_sections_sink"))
    rows2, tol = e2["rows"], e2["tol"]
    c1, c2, c3 = chart_outliers(e3), chart_dims(rows2, tol), chart_instr(e4)

    # headline numbers
    pc128 = next(r for r in rows2 if r["d"] == 128 and r["family"] == "outlier" and r["scenario"] == "prefix caching")
    pk = [r for r in e1 if r["d"] == 128 and r["scenario"] == "sequence packing"][0]
    plain = [r for r in e1 if r["d"] == 128 and r["scenario"] == "plain causal"][0]
    iden = all(r["bit_identical"] for r in e5["rows"])
    unchanged = [r for r in rows2 if r["d"] in (64, 256) and r["before_worst"] == r["after_worst"]]
    n_pow2 = len([r for r in rows2 if r["d"] in (64, 256)])
    s_hi = next(r for r in e4 if r["L"] == 512)
    s_lo = next(r for r in e4 if r["L"] == 4096)
    tiles = [
        (f"{pc128['before_pass']}/{pc128['n_seeds']} &rarr; {pc128['after_pass']}/{pc128['n_seeds']}",
         "seeds within tolerance, prefix caching d=128 with outlier inputs"),
        ("fails &rarr; runs" if (not pk["before"]["runs"] and pk["after"]["runs"]) else "n/a",
         "sequence packing with scale 1/sqrt(128)"),
        (f"-{s_lo['reduction']:.0%} to -{s_hi['reduction']:.0%}", "MM1 evict/mask instructions, 512&ndash;4K tokens (model)"),
        ("identical" if iden else "DIFFERS", "multi-section flash + sink + out_neg_max vs upstream (emulation)"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="big">{a}</div><div>{b}</div></div>' for a, b in tiles)

    seeds = len(e2["seeds"])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>attention_cte scale patch</title>
<style>
:root{{--page:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--ok:#006300;--bad:#d03b3b}}
@media(prefers-color-scheme:dark){{:root{{--page:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--ok:#0ca30c;--bad:#ec835a}}}}
body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:24px 16px 56px}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:0 0 6px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}
.lede,.sub{{color:var(--ink2)}}.sub{{font-size:12.5px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}.big{{font-size:26px;font-weight:600}}
img{{max-width:100%;height:auto;background:#fff;border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}}
.scroll{{overflow-x:auto}}.b{{font-weight:600;white-space:nowrap}}.b.ok{{color:var(--ok)}}.b.bad{{color:var(--bad)}}
code{{background:var(--page);padding:1px 5px;border-radius:4px}}.pend{{color:var(--ink2)}}
</style></head><body><main>
<h1>attention_cte: non-power-of-two softmax scale, before vs after the patch</h1>
<p class="lede">Patch <code>attention_cte_fp32_scale_in_exp.patch</code> on base <code>ad2f1b9</code>. <b>Sections 1&ndash;5:</b> CPU emulation (numpy, bf16) of the kernel's arithmetic and decision logic. <b>Section 6:</b> measured on Trainium2 hardware (trn2.48xlarge), running the patch's new test cases against both the original and the patched kernel.</p>
<div class="tiles">{tiles_html}</div>

<div class="card"><h2>1. Which path each config takes</h2>
<div class="sub">The kernel's scale decision code (before and after the patch) evaluated per configuration, with scale = 1/sqrt(d).</div>
{paths_table(e1)}</div>

<div class="card"><h2>2. Accuracy vs outlier strength</h2>
<div class="sub">Worst of {seeds} seeds, S = {e3['S']}. Error = max|out &minus; ref| / max|ref| against float64 attention on the same bf16 inputs. Outlier inputs are a stylised model, not measured activations.</div>
{img(c1, "error vs outlier gain")}</div>

<div class="card"><h2>3. Accuracy by head dimension</h2>
<div class="sub">Power-of-two scales (d = 64, 256) are unchanged by the patch: {len(unchanged)} of {n_pow2} cases give identical error before and after.</div>
{img(c2, "error by head dim")}
<details><summary>Full table (d = 64, 128, 256; all input families and scenarios)</summary>{acc_table(rows2, tol)}</details></div>

<div class="card"><h2>4. Instruction count on the causal diagonal</h2>
<div class="sub">128-row Q groups, 512-key tiles, tiles above the diagonal skipped. A diagonal tile goes from 3 instructions to 1. This is a count, not a timing; wall-clock gain depends on whether this stage is on the critical path.</div>
{img(c3, "instruction model")}
{table(["Prompt", "Full tiles", "Diagonal tiles", "Diagonal share", "Before", "After", "Reduction", "GpSimd affine_select elements removed"],
       [[r["L"], r["full"], r["diag"], f"{r['diag_share']:.0%}", r["before"], r["after"], f"-{r['reduction']:.0%}", f"{r['affine_elems_removed'] / 1e6:.2f} M"] for r in e4])}</div>

<div class="card"><h2>5. Flash attention sections, sink, out_neg_max</h2>
<div class="sub">d = 128, Sq = {e5['sq']}, Skv = {e5['skv']}. Patch-style and upstream-style section loops compared bit for bit. This only shows the algebra is consistent in emulation; on hardware a fused multiply-add could differ in the last fp32 bit.</div>
{table(["Sections", "Sink", "max |out diff|", "max |neg_max diff|", "Bit-identical"],
       [[r["sections"], "none" if r["sink"] is None else r["sink"], r["max_out_diff"], r["max_negmax_diff"], badge(r["bit_identical"], "yes" if r["bit_identical"] else "no")] for r in e5["rows"]])}</div>

{device_section()}
</main></body></html>"""
    out = os.path.join(HERE, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print("wrote", out)


if __name__ == "__main__":
    main()

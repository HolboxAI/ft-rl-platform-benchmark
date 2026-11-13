#!/usr/bin/env python3
"""Train reward per step, TPU run vs GPU run, as SVG for the README (light + dark).

GitHub READMEs show images, not scripts, so this writes two static SVGs (one per GitHub
theme) that the README embeds with <picture>. Standard library only.

    python analysis/reward_chart.py logs/fetched/<run_id>   # -> docs/img/reward_per_step_{light,dark}.svg

Colors: the validated categorical slots 1 (blue, GPU) and 2 (orange, TPU) in each mode's steps.
"""
from __future__ import annotations

import json
import os
import sys
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

THEMES = {
    "light": {"gpu": "#2a78d6", "tpu": "#eb6834", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "ring": "#ffffff"},
    "dark": {"gpu": "#3987e5", "tpu": "#d95926", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "ring": "#0d1117"},
}


def load(run_dir):
    tpu = {}
    for line in open(os.path.join(run_dir, "metrics.jsonl")):
        r = json.loads(line)
        v = r["data"].get("reward/avg_raw_reward")
        if v is not None and r["step"] >= 1:
            tpu[int(r["step"])] = float(v)
    gpu_curve = json.load(open(os.path.join(ROOT, "config", "gpu_reference_final_results.json")))
    gpu = {i + 1: float(v) for i, v in enumerate(gpu_curve["experiments"]["cholec_rl"]["reward_curve"])}
    return sorted(tpu.items()), sorted(gpu.items())


def svg(tpu, gpu, t):
    W, H = 760, 330
    ml, mr, mt, mb = 48, 24, 44, 40
    x0, x1, y0, y1 = 1, 16, 0.0, 0.4
    X = lambda s: ml + (s - x0) / (x1 - x0) * (W - ml - mr)
    Y = lambda v: mt + (1 - (v - y0) / (y1 - y0)) * (H - mt - mb)
    font = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" {font}>',
         "<title>Train reward per step: TPU vs GPU</title>",
         "<desc>TPU run: 0.292 at step 1, then 0.0 for steps 2 to 16. GPU run: 0.09 to 0.31 for steps 1 to 13, "
         "then 0.0 at steps 14 and 15.</desc>"]
    # legend (line keys), above the plot
    lx = ml
    for key, label in (("gpu", "GPU run, 4× A10G"), ("tpu", "TPU run, v6e-8")):
        o.append(f'<line x1="{lx}" y1="18" x2="{lx + 18}" y2="18" stroke="{t[key]}" stroke-width="2" stroke-linecap="round"/>')
        o.append(f'<text x="{lx + 24}" y="22" font-size="13" fill="{t["text2"]}">{escape(label)}</text>')
        lx += 24 + 8 * len(label) + 28
    # grid + y ticks
    for v in (0.0, 0.1, 0.2, 0.3, 0.4):
        y = Y(v)
        o.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W - mr}" y2="{y:.1f}" stroke="{t["axis"] if v == 0 else t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{t["muted"]}" '
                 f'style="font-variant-numeric:tabular-nums">{v:.1f}</text>')
    for s in range(1, 17):
        o.append(f'<text x="{X(s):.1f}" y="{H - mb + 16}" font-size="11" text-anchor="middle" fill="{t["muted"]}">{s}</text>')
    o.append(f'<text x="{W - mr}" y="{H - 6}" font-size="11" text-anchor="end" fill="{t["muted"]}">training step</text>')
    # series
    for key, pts in (("tpu", tpu), ("gpu", gpu)):  # GPU last: its zero segment stays visible over TPU's
        d = " ".join(f'{"M" if i == 0 else "L"}{X(s):.1f},{Y(v):.1f}' for i, (s, v) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{t[key]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        s, v = pts[-1]
        o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="{t[key]}" stroke="{t["ring"]}" stroke-width="2"/>')
    # selective direct labels: TPU's only non-zero point, the GPU peak, the GPU collapse
    s1, v1 = tpu[0]
    o.append(f'<circle cx="{X(s1):.1f}" cy="{Y(v1):.1f}" r="4" fill="{t["tpu"]}" stroke="{t["ring"]}" stroke-width="2"/>')
    o.append(f'<text x="{X(s1) + 9:.1f}" y="{Y(v1) - 8:.1f}" font-size="12" fill="{t["text"]}">'
             f'<tspan font-weight="600">{v1:.3f}</tspan><tspan fill="{t["text2"]}"> TPU, step 1 — then 0.0</tspan></text>')
    sp, vp = max(gpu, key=lambda p: p[1])
    o.append(f'<circle cx="{X(sp):.1f}" cy="{Y(vp):.1f}" r="4" fill="{t["gpu"]}" stroke="{t["ring"]}" stroke-width="2"/>')
    o.append(f'<text x="{X(sp):.1f}" y="{Y(vp) - 10:.1f}" font-size="12" text-anchor="middle" fill="{t["text"]}">'
             f'<tspan font-weight="600">{vp:.3f}</tspan><tspan fill="{t["text2"]}"> GPU peak</tspan></text>')
    sc = next(s for s, v in gpu if s > sp and v == 0.0)
    o.append(f'<text x="{X(sc) + 6:.1f}" y="{Y(0) - 22:.1f}" font-size="12" fill="{t["text2"]}">GPU collapse</text>')
    o.append("</svg>")
    return "\n".join(o)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "logs", "fetched", "cholect50-skyrl-tpu-20261024-121826")
    tpu, gpu = load(run_dir)
    out_dir = os.path.join(ROOT, "docs", "img")
    os.makedirs(out_dir, exist_ok=True)
    for name, theme in THEMES.items():
        p = os.path.join(out_dir, f"reward_per_step_{name}.svg")
        with open(p, "w") as f:
            f.write(svg(tpu, gpu, theme))
        print("wrote", p)


if __name__ == "__main__":
    main()

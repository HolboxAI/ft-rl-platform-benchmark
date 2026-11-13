#!/usr/bin/env python3
"""Charts of a DS-1000 TPU run, as static SVGs (one per GitHub theme), standard library only.

    python skyrl-tpu/skyrl_ds1000/analysis/charts.py logs/<run_id>/metrics.jsonl   # -> skyrl_ds1000/docs/img/

Reads the run's metrics.jsonl (one SkyRL tracker.log call per line). Charts:

* eval_pass_rate   val pass@1 (206 problems, greedy) before training and at each eval
* train_reward     train reward (mean of 128 samples) and pass@8 per step
* training_health  KL to the reference, mean reply length, step time (small multiples, one scale each)

Colors and marks as miles-tpu/analysis/charts.py: validated categorical slots 1 (blue) and 2 (orange) in each mode's
own steps. GitHub shows images, not scripts, so there is no hover layer: the README carries the numbers as tables.
"""
from __future__ import annotations

import json
import os
import sys
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "docs", "img")
FONT = 'font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"'
STEPS = 20  # planned training steps (x axis runs 0..STEPS so a partial run shows how far it got)

THEMES = {
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "ring": "#ffffff"},
    "dark": {"s1": "#3987e5", "s2": "#d95926", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "ring": "#0d1117"},
}


def load(path):
    evals, train = [], {}
    for line in open(path):
        r = json.loads(line)
        s, d = int(r["step"]), r["data"]
        if "eval/all/pass_at_1" in d:
            evals.append((s, float(d["eval/all/pass_at_1"])))
        if "reward/avg_raw_reward" in d:
            train[s] = {
                "reward": float(d["reward/avg_raw_reward"]),
                "pass8": float(d.get("reward/avg_pass_at_8", "nan")),
                "kl": float(d.get("policy/policy_kl", "nan")),
                "tokens": float(d.get("generate/avg_num_tokens", "nan")),
                "minutes": float(d.get("timing/step", "nan")) / 60,
            }
    return evals, [(s, train[s]) for s in sorted(train)]


class Plot:
    """One plot area: data → pixel mapping plus the axis, grid and series primitives."""

    def __init__(self, *, x0, y0, w, h, steps, ymin, ymax, yticks, fmt="{:.1f}"):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.s0, self.s1 = steps
        self.ymin, self.ymax, self.yticks, self.fmt = ymin, ymax, yticks, fmt

    def X(self, s):
        return self.x0 + (s - self.s0) / (self.s1 - self.s0) * self.w

    def Y(self, v):
        return self.y0 + (1 - (v - self.ymin) / (self.ymax - self.ymin)) * self.h

    def grid(self, t, *, xticks):
        o = []
        for v in self.yticks:
            y = self.Y(v)
            stroke = t["axis"] if v == self.ymin else t["grid"]
            o.append(f'<line x1="{self.x0}" y1="{y:.1f}" x2="{self.x0 + self.w}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1"/>')
            o.append(f'<text x="{self.x0 - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{t["muted"]}" '
                     f'style="font-variant-numeric:tabular-nums">{self.fmt.format(v)}</text>')
        for s in xticks:
            o.append(f'<text x="{self.X(s):.1f}" y="{self.y0 + self.h + 16}" font-size="11" text-anchor="middle" '
                     f'fill="{t["muted"]}">{s}</text>')
        return o

    def line(self, pts, color):
        d = " ".join(f'{"M" if i == 0 else "L"}{self.X(s):.1f},{self.Y(v):.1f}' for i, (s, v) in enumerate(pts))
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'

    def dot(self, s, v, color, t, r=4):
        return f'<circle cx="{self.X(s):.1f}" cy="{self.Y(v):.1f}" r="{r}" fill="{color}" stroke="{t["ring"]}" stroke-width="2"/>'


def legend(t, x, y, items):
    o = []
    for color, label in items:
        o.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 18}" y2="{y - 4}" stroke="{t[color]}" stroke-width="2" stroke-linecap="round"/>')
        o.append(f'<text x="{x + 24}" y="{y}" font-size="13" fill="{t["text2"]}">{escape(label)}</text>')
        x += 24 + 7.2 * len(label) + 28
    return o


def svg(w, h, title, desc, body):
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" {FONT}>',
                      f"<title>{escape(title)}</title>", f"<desc>{escape(desc)}</desc>", *body, "</svg>"])


def eval_pass_rate(evals, train, t):
    W, H = 760, 300
    p = Plot(x0=48, y0=40, w=W - 48 - 40, h=H - 40 - 44, steps=(0, STEPS), ymin=0.0, ymax=0.6,
             yticks=(0.0, 0.2, 0.4, 0.6), fmt="{:.0%}")
    o = [f'<text x="{p.x0}" y="20" font-size="12" fill="{t["text2"]}">Val pass@1, 206 DS-1000 problems, greedy '
         f'(Qwen3.5-4B; step 0 = before training)</text>']
    o += p.grid(t, xticks=range(0, STEPS + 1, 2))
    o.append(f'<text x="{W - 24}" y="{H - 8}" font-size="11" text-anchor="end" fill="{t["muted"]}">training step</text>')
    o.append(p.line(evals, t["s1"]))
    base = evals[0][1]
    for s, v in evals:
        o.append(p.dot(s, v, t["s1"], t, r=5))
        extra = "" if s == 0 else f" ({(v - base) * 100:+.1f} pts)"
        anchor = "end" if s == STEPS else "middle"  # the last point's label would run past the plot edge
        o.append(f'<text x="{p.X(s) + (4 if s == STEPS else 0):.1f}" y="{p.Y(v) - 12:.1f}" font-size="12" '
                 f'text-anchor="{anchor}" fill="{t["text"]}">'
                 f'<tspan font-weight="600">{v:.1%}</tspan><tspan fill="{t["text2"]}">{escape(extra)}</tspan></text>')
    if train and train[-1][0] < STEPS:
        o.append(f'<text x="{p.X(STEPS):.1f}" y="{p.Y(0.02):.1f}" font-size="11" text-anchor="end" fill="{t["muted"]}">'
                 f'run in progress: {train[-1][0]} of {STEPS} steps done</text>')
    desc = "Val pass@1: " + ", ".join(f"step {s}: {v:.1%}" for s, v in evals) + "."
    return svg(W, H, "Val pass@1 before and during training", desc, o)


def train_reward(evals, train, t):
    W, H = 760, 340
    p = Plot(x0=48, y0=48, w=W - 48 - 40, h=H - 48 - 44, steps=(0, STEPS), ymin=0.0, ymax=1.0,
             yticks=(0.0, 0.25, 0.5, 0.75, 1.0), fmt="{:.2f}")
    R = [(s, m["reward"]) for s, m in train]
    P8 = [(s, m["pass8"]) for s, m in train]
    o = legend(t, p.x0, 22, (("s1", "Train reward (mean of 16 problems × 8 samples)"), ("s2", "pass@8 (share of problems with ≥1 pass)")))
    o += p.grid(t, xticks=range(0, STEPS + 1, 2))
    o.append(f'<text x="{W - 24}" y="{H - 8}" font-size="11" text-anchor="end" fill="{t["muted"]}">training step</text>')
    o.append(p.line(P8, t["s2"]))
    o.append(p.line(R, t["s1"]))
    for pts, key in ((R, "s1"), (P8, "s2")):
        for s, v in (pts[0], pts[-1]):
            o.append(p.dot(s, v, t[key], t, r=3.5))
    # selective direct labels: first and last value of each series
    for pts, dy in ((R, 18), (P8, -10)):
        for s, v in (pts[0], pts[-1]):
            o.append(f'<text x="{p.X(s):.1f}" y="{p.Y(v) + dy:.1f}" font-size="12" text-anchor="middle" '
                     f'font-weight="600" fill="{t["text"]}">{v:.2f}</text>')
    first5 = sum(v for _, v in R[:5]) / len(R[:5])
    last5 = sum(v for _, v in R[-5:]) / len(R[-5:])
    o.append(f'<text x="{W - 40}" y="{p.y0 + 14}" font-size="12" text-anchor="end" fill="{t["text2"]}">'
             f'mean reward, steps {R[0][0]}–{R[min(4, len(R) - 1)][0]}: <tspan font-weight="600" fill="{t["text"]}">{first5:.3f}</tspan>'
             f'   steps {R[-5][0] if len(R) >= 5 else R[0][0]}–{R[-1][0]}: <tspan font-weight="600" fill="{t["text"]}">{last5:.3f}</tspan></text>')
    desc = (f"Train reward {R[0][1]:.3f} at step {R[0][0]}, {R[-1][1]:.3f} at step {R[-1][0]}; mean of the first five steps "
            f"{first5:.3f}, of the last five {last5:.3f}. pass@8 {P8[0][1]:.2f} to {P8[-1][1]:.2f}.")
    return svg(W, H, "Train reward and pass@8 per step", desc, o)


def _nice_max(v):
    """Smallest of 1, 2, 2.5, 5 × 10^k at or above 1.15 × v, so the top value keeps some headroom."""
    import math

    v *= 1.15
    k = 10 ** math.floor(math.log10(v))
    return next(m * k for m in (1, 2, 2.5, 5, 10) if m * k >= v)


def training_health(evals, train, t):
    W, H = 760, 260
    panels = (("kl", "KL to reference", "{:.3f}", "{:.3f}"),
              ("tokens", "Mean reply length, tokens (limit 512)", "{:.0f}", "{:.0f}"),
              ("minutes", "Step time, minutes", "{:.0f}", "{:.1f}"))
    pw, gap, top = 180, 76, 44   # 48 + 3*pw + 2*gap = 740 <= W - 20
    o = []
    for i, (key, title, fmt, vfmt) in enumerate(panels):
        cx = 48 + i * (pw + gap)
        ymax = _nice_max(max(m[key] for _, m in train))
        p = Plot(x0=cx, y0=top, w=pw, h=H - top - 50, steps=(0, STEPS), ymin=0, ymax=ymax,
                 yticks=(0, ymax / 2, ymax), fmt=fmt)
        o.append(f'<text x="{cx}" y="22" font-size="13" font-weight="600" fill="{t["text"]}">{escape(title)}</text>')
        o += p.grid(t, xticks=(0, 10, 20))
        pts = [(s, m[key]) for s, m in train]
        o.append(p.line(pts, t["s1"]))
        s, v = pts[-1]
        o.append(p.dot(s, v, t["s1"], t, r=3.5))
        o.append(f'<text x="{p.X(s):.1f}" y="{p.Y(v) - 10:.1f}" font-size="11" text-anchor="middle" font-weight="600" '
                 f'fill="{t["text"]}">{vfmt.format(v)}</text>')
    o.append(f'<text x="{W - 24}" y="{H - 8}" font-size="11" text-anchor="end" fill="{t["muted"]}">training step (value at the last step)</text>')
    desc = "; ".join(f"{title}: {train[0][1][key]:.3f} at step {train[0][0]}, {train[-1][1][key]:.3f} at step {train[-1][0]}"
                     for key, title, *_ in panels)
    return svg(W, H, "Training health: KL, reply length, step time", desc, o)


def main():
    evals, train = load(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("eval_pass_rate", eval_pass_rate), ("train_reward", train_reward), ("training_health", training_health)):
        for theme, t in THEMES.items():
            path = os.path.join(OUT, f"{name}_{theme}.svg")
            with open(path, "w") as f:
                f.write(fn(evals, train, t))
            print("wrote", path)


if __name__ == "__main__":
    main()

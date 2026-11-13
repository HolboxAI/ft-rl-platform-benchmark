#!/usr/bin/env python3
"""Build the report for a SkyRL CholecT50 TPU run: one self-contained HTML file.

Reads the run's ``metrics.jsonl`` (written by ``skyrl_cholect50/entrypoint.py``:
one line per ``tracker.log`` call) and the GPU run's results
(``config/gpu_reference_final_results.json``, copied from
``ft-rl-platform-benchmark-git/skyrl/metrics/final_results.json``), and writes:

* ``report.html``  stat tiles (val reward before/after, TPU vs GPU), training
  reward per step (TPU vs GPU), where each TPU step's time goes, training
  diagnostics as small multiples, and a table of every plotted value;
* ``final_results.json``  the TPU run in the GPU run's schema, for the benchmark tables.

    python analysis/plot_run.py logs/<run_id>              # -> logs/<run_id>/report.html
    python analysis/plot_run.py --demo /tmp/demo            # synthetic data, to check rendering only

No third-party packages; the HTML has no external requests.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPU_REF = os.path.join(ROOT, "config", "gpu_reference_final_results.json")

TIMING_PARTS = [  # (metric key, label) in stack order, bottom to top
    ("timing/generate", "Generation"),
    ("timing/fwd_logprobs_values_reward", "Old + ref logprobs"),
    ("timing/policy_train", "Policy update"),
    ("timing/sync_weights", "Weight sync"),
]
DIAGNOSTICS = [  # (metric key or regex, title, unit)
    (r"reward/avg_pass_at_\d+", "Pass@8 (train rollouts)", ""),
    ("loss/avg_kl", "KL to reference", ""),
    ("policy/policy_loss", "Policy loss", ""),
    ("generate/avg_num_tokens", "Mean response length", "tokens"),
    ("policy/grad_norm", "Gradient norm", ""),
    ("policy/entropy", "Entropy", ""),
]
TRAIN_REWARD = "reward/avg_raw_reward"
EVAL_SCORE = "eval/all/avg_score"


# ------------------------------------------------------------------ loading
def load_metrics(path):
    by_step = defaultdict(dict)
    times = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = int(rec["step"])
            by_step[step].update(rec.get("data") or {})
            times.setdefault(step, rec.get("time"))
    return dict(sorted(by_step.items())), times


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def find_key(keys, pattern):
    if pattern in keys:
        return pattern
    rx = re.compile(pattern + "$")
    for k in sorted(keys):
        if rx.match(k):
            return k
    return None


def read_env(run_dir):
    p = os.path.join(run_dir, "environment.txt")
    if not os.path.exists(p):
        return {}
    env = {}
    text = open(p).read()
    for k, rx in [("skyrl", r"skyrl=(.+)"), ("accelerator", r"^(v\d+\w*-\d+)\s*$"), ("torch", r"torch (\S+) transformers"),
                  ("transformers", r"transformers (\S+)"), ("torch_tpu", r"torch_tpu (\S+)"), ("engine", r'"engine": "(\w+)"')]:
        m = re.search(rx, text, re.M)
        if m:
            env[k] = m.group(1).strip()
    return env


def build(run_dir, metrics_path, gpu_ref_path, demo=False):
    steps, times = load_metrics(metrics_path)
    keys = set(k for d in steps.values() for k in d)
    gpu = json.load(open(gpu_ref_path))["experiments"]["cholec_rl"]

    train_steps = [s for s in steps if s >= 1 and num(steps[s].get(TRAIN_REWARD)) is not None]
    tpu_reward = [[s, num(steps[s][TRAIN_REWARD])] for s in train_steps]
    gpu_reward = [[i + 1, v] for i, v in enumerate(gpu.get("reward_curve") or [])]

    evals = [[s, num(d[EVAL_SCORE])] for s, d in steps.items() if num(d.get(EVAL_SCORE)) is not None]
    eval_before = next((v for s, v in evals if s == 0), None)
    eval_after = next((v for s, v in reversed(evals) if s > 0), None)

    timing_rows = []
    for s in train_steps:
        d = steps[s]
        vals = {label: num(d.get(k)) for k, label in TIMING_PARTS}
        total = num(d.get("timing/step"))
        parts = sum(v for v in vals.values() if v is not None)
        if total is not None and total > parts + 0.5:
            vals["Other"] = total - parts
        timing_rows.append({"x": s, "values": {k: v for k, v in vals.items() if v is not None}})
    timing_cats = [label for _, label in TIMING_PARTS if any(label in r["values"] for r in timing_rows)]
    if any("Other" in r["values"] for r in timing_rows):
        timing_cats.append("Other")

    diags = []
    for pat, title, unit in DIAGNOSTICS:
        k = find_key(keys, pat)
        if k is None:
            continue
        pts = [[s, num(steps[s].get(k))] for s in train_steps if num(steps[s].get(k)) is not None]
        if pts:
            diags.append({"key": k, "title": title, "unit": unit, "points": pts})

    step_times = [num(steps[s].get("timing/step")) for s in train_steps]
    step_times = [t for t in step_times if t is not None]
    env = read_env(run_dir)
    expected_steps = 16
    data = {
        "demo": demo,
        "run_id": os.path.basename(os.path.abspath(run_dir)),
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "env": env,
        "gpu": {"label": "GPU · 4× A10G (g5.12xlarge)", "eval_before": gpu.get("start_score"),
                "eval_after": gpu.get("final_score"), "steps": gpu.get("steps"), "wall_time_s": gpu.get("wall_time_s"),
                "reward": gpu_reward},
        "tpu": {"label": f"TPU · {env.get('accelerator', 'Cloud TPU')}", "eval_before": eval_before, "eval_after": eval_after,
                "steps": len(train_steps), "expected_steps": expected_steps, "reward": tpu_reward,
                "mean_step_s": sum(step_times) / len(step_times) if step_times else None,
                "wall_time_s": (max(times.values()) - min(times.values())) if len(times) > 1 else None},
        "timing": {"categories": timing_cats, "rows": timing_rows},
        "diagnostics": diags,
        "evals": evals,
    }
    return data


def final_results(data):
    t = data["tpu"]
    return {
        "platform": "skyrl-tpu",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiments": {"cholec_rl": {
            "experiment": "cholec_t50_grpo_rlvr",
            "method": "grpo",
            "job_id": data["run_id"],
            "state": "completed" if t["steps"] >= t["expected_steps"] else "partial",
            "base_model": "Qwen/Qwen3.5-4B",
            "hardware": data["env"].get("accelerator"),
            "skyrl": data["env"].get("skyrl"),
            "rollout_engine": data["env"].get("engine"),
            "learning_rate": 1e-5, "max_context_length": 4096, "epochs": 1, "dataset_size": 256,
            "steps": t["steps"], "wall_time_s": t["wall_time_s"], "mean_step_s": t["mean_step_s"],
            "start_score": t["eval_before"], "final_score": t["eval_after"],
            "reward_curve": [v for _, v in t["reward"]],
            "synthetic_demo_data": data["demo"],
        }},
    }


# ------------------------------------------------------------------ demo data
def write_demo_metrics(path):
    """Synthetic metrics with the real keys, ONLY to check that the report renders."""
    rnd = random.Random(7)
    with open(path, "w") as f:
        def log(step, data):
            f.write(json.dumps({"step": step, "time": 1_700_000_000 + step * 600, "data": data}) + "\n")
        log(0, {EVAL_SCORE: 0.31, "eval/all/pass_at_1": 0.31, "timing/eval": 210.0})
        r = 0.12
        for s in range(1, 17):
            r = max(0.0, min(0.6, r + rnd.uniform(-0.05, 0.07)))
            gen, fwd, tr, sync = rnd.uniform(150, 190), rnd.uniform(40, 55), rnd.uniform(90, 120), rnd.uniform(70, 95)
            log(s, {TRAIN_REWARD: r, "reward/avg_pass_at_8": min(1, r * 2.2), "loss/avg_kl": 0.002 * s * rnd.uniform(0.8, 1.2),
                    "policy/policy_loss": rnd.uniform(-0.05, 0.05), "generate/avg_num_tokens": rnd.uniform(28, 44),
                    "policy/grad_norm": rnd.uniform(0.5, 2.0), "timing/generate": gen, "timing/fwd_logprobs_values_reward": fwd,
                    "timing/policy_train": tr, "timing/sync_weights": sync, "timing/step": gen + fwd + tr + sync + rnd.uniform(5, 15)})
        log(16, {EVAL_SCORE: 0.36, "timing/eval": 205.0})


# ------------------------------------------------------------------ HTML
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SkyRL CholecT50 TPU</title>
<style>
:root{color-scheme:light;
 --page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
 --series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;
 --warn-bg:#fff4dc;--warn-ink:#6b4a00;--good:#006300;--bad:#b42323}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
 --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;
 --warn-bg:#3a2c08;--warn-ink:#f3d58a;--good:#0ca30c;--bad:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;
 --page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
 --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;
 --warn-bg:#3a2c08;--warn-ink:#f3d58a;--good:#0ca30c;--bad:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;font-weight:600;margin:0 0 4px}
h2{font-size:15px;font-weight:600;margin:0 0 2px}
.sub{color:var(--text-secondary);margin:0 0 16px}
.muted{color:var(--text-muted)}
.banner{background:var(--warn-bg);color:var(--warn-ink);border-radius:8px;padding:10px 14px;margin:0 0 16px;font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:0 0 16px}
.tile,.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .label{color:var(--text-secondary);font-size:13px}
.tile .value{font-size:28px;font-weight:600;margin:2px 0}
.tile .cmp{color:var(--text-secondary);font-size:13px}
.card{margin:0 0 16px}
.card .desc{color:var(--text-secondary);font-size:13px;margin:0 0 8px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:4px 0 6px;color:var(--text-secondary);font-size:13px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.key-line{width:16px;height:2px;border-radius:1px;display:inline-block}
.key-box{width:10px;height:10px;border-radius:2px;display:inline-block}
.chart{position:relative;width:100%}
.chart svg{display:block;width:100%;overflow:visible}
.multiples{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.multiples .card{margin:0}
.tip{position:fixed;pointer-events:none;z-index:10;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);
 border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;min-width:150px}
.tip .t{color:var(--text-secondary);margin-bottom:4px}
.tip .row{display:flex;align-items:center;gap:8px;justify-content:space-between}
.tip .row b{font-variant-numeric:tabular-nums}
.tip .row .n{display:inline-flex;align-items:center;gap:6px;color:var(--text-secondary)}
details{margin-top:8px}
summary{cursor:pointer;color:var(--text-secondary)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;font-size:12px;margin-top:8px;min-width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600}
.empty{color:var(--text-muted);padding:24px 0;text-align:center}
</style></head>
<body><main>
<div id="app"></div>
<div class="tip" id="tip" role="status" aria-live="polite"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const NS = "http://www.w3.org/2000/svg";
const app = document.getElementById("app");
const tip = document.getElementById("tip");

function h(tag, attrs, ...kids){ const e=document.createElement(tag); for(const [k,v] of Object.entries(attrs||{})){ if(k==="class") e.className=v; else e.setAttribute(k,v);} for(const c of kids){ if(c==null) continue; e.append(c instanceof Node?c:document.createTextNode(String(c))); } return e; }
function s(tag, attrs){ const e=document.createElementNS(NS,tag); for(const [k,v] of Object.entries(attrs||{})) e.setAttribute(k,v); return e; }
function txt(x,y,str,attrs){ const t=s("text",Object.assign({x,y,fill:"var(--text-muted)","font-size":11},attrs||{})); t.textContent=str; return t; }
const f3 = v => v==null ? "—" : (Math.abs(v)>=100 ? v.toFixed(0) : Math.abs(v)>=10 ? v.toFixed(1) : v.toFixed(3));
const fsec = v => v==null ? "—" : v>=3600 ? (v/3600).toFixed(1)+" h" : v>=60 ? (v/60).toFixed(1)+" min" : v.toFixed(0)+" s";
function niceTicks(lo, hi, n){ if(hi<=lo) hi=lo+1; const span=hi-lo, step0=span/Math.max(1,n); const mag=Math.pow(10,Math.floor(Math.log10(step0))); const step=[1,2,2.5,5,10].map(m=>m*mag).find(st=>span/st<=n)||10*mag; const out=[]; let v=Math.floor(lo/step)*step; out.push(+v.toFixed(10)); while(v<hi-1e-9){ v+=step; out.push(+v.toFixed(10)); } return out; }

function showTip(ev, title, rows){
  tip.replaceChildren(h("div",{class:"t"},title));
  for(const r of rows){ const key=h("i",{class:r.box?"key-box":"key-line"}); key.style.background=r.color; tip.append(h("div",{class:"row"},h("span",{class:"n"},key,r.name),h("b",{},r.value))); }
  tip.style.display="block";
  const pad=14, w=tip.offsetWidth, hgt=tip.offsetHeight;
  let x=ev.clientX+pad, y=ev.clientY+pad;
  if(x+w>window.innerWidth-8) x=ev.clientX-w-pad;
  if(y+hgt>window.innerHeight-8) y=ev.clientY-hgt-pad;
  tip.style.left=x+"px"; tip.style.top=y+"px";
}
function hideTip(){ tip.style.display="none"; }

function legend(items, box){ return h("div",{class:"legend"},...items.map(it=>{ const k=h("i",{class:box?"key-box":"key-line"}); k.style.background=it.color; return h("span",{},k,it.name); })); }

// ------------------------------------------------------------ line chart
function lineChart(el, opt){
  const W=Math.max(280, el.clientWidth), H=opt.height||260;
  const m={l:44, r:opt.endLabels?92:16, t:12, b:30};
  const all=opt.series.flatMap(sr=>sr.points);
  el.replaceChildren();
  if(!all.length){ el.append(h("div",{class:"empty"},"No data yet")); return; }
  const xs=all.map(p=>p[0]), ys=all.map(p=>p[1]);
  const x0=opt.xMin??Math.min(...xs), x1=opt.xMax??Math.max(...xs);
  const yT=niceTicks(opt.yMin??Math.min(0,...ys), Math.max(...ys)*1.08 || 1, 4), y0=yT[0], y1=yT[yT.length-1];
  const X=v=>m.l+(x1===x0?0.5:(v-x0)/(x1-x0))*(W-m.l-m.r), Y=v=>m.t+(1-(v-y0)/(y1-y0))*(H-m.t-m.b);
  const svg=s("svg",{viewBox:`0 0 ${W} ${H}`,height:H,role:"img","aria-label":opt.aria||""});
  for(const t of yT){ svg.append(s("line",{x1:m.l,x2:W-m.r,y1:Y(t),y2:Y(t),stroke:t===y0?"var(--axis)":"var(--grid)","stroke-width":1})); svg.append(txt(m.l-6,Y(t)+4,f3(t),{"text-anchor":"end"})); }
  const xStep=Math.max(1,Math.ceil((x1-x0+1)/Math.max(2,Math.floor((W-m.l-m.r)/34))));
  for(let v=Math.ceil(x0); v<=x1; v+=xStep) svg.append(txt(X(v),H-m.b+16,String(v),{"text-anchor":"middle"}));
  if(opt.xLabel) svg.append(txt(W-m.r,H-2,opt.xLabel,{"text-anchor":"end"}));
  for(const sr of opt.series){
    if(!sr.points.length) continue;
    const d=sr.points.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join("");
    svg.append(s("path",{d,fill:"none",stroke:sr.color,"stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"}));
    const last=sr.points[sr.points.length-1];
    svg.append(s("circle",{cx:X(last[0]),cy:Y(last[1]),r:4,fill:sr.color,stroke:"var(--surface-1)","stroke-width":2}));
    sr._end=[X(last[0]),Y(last[1]),last[1]];
  }
  if(opt.endLabels){ // value + name at line ends; push apart only if they would overlap, with a leader
    const ends=opt.series.filter(sr=>sr._end).map(sr=>({sr,y:sr._end[1]})).sort((a,b)=>a.y-b.y);
    for(let i=1;i<ends.length;i++) if(ends[i].y-ends[i-1].y<26) ends[i].y=ends[i-1].y+26;
    for(const e of ends){ const [ex,ey,v]=e.sr._end; const lx=W-m.r+10;
      if(Math.abs(e.y-ey)>2) svg.append(s("line",{x1:ex+5,y1:ey,x2:lx-3,y2:e.y,stroke:"var(--axis)","stroke-width":1}));
      svg.append(txt(lx,e.y-1,f3(v),{fill:"var(--text-primary)","font-size":12,"font-weight":600}));
      svg.append(txt(lx,e.y+12,e.sr.short||e.sr.name,{fill:"var(--text-secondary)"})); }
  }
  const cross=s("line",{y1:m.t,y2:H-m.b,stroke:"var(--axis)","stroke-width":1,visibility:"hidden"}); svg.append(cross);
  const hit=s("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:"transparent",tabindex:0}); svg.append(hit);
  const xsAll=[...new Set(xs)].sort((a,b)=>a-b);
  function at(ev){ const r=svg.getBoundingClientRect(); const px=(ev.clientX-r.left)*W/r.width; let best=xsAll[0]; for(const v of xsAll) if(Math.abs(X(v)-px)<Math.abs(X(best)-px)) best=v;
    cross.setAttribute("x1",X(best)); cross.setAttribute("x2",X(best)); cross.setAttribute("visibility","visible");
    showTip(ev,(opt.xName||"Step")+" "+best, opt.series.map(sr=>{ const p=sr.points.find(q=>q[0]===best); return {name:sr.name,color:sr.color,value:p?f3(p[1])+(opt.unit?" "+opt.unit:""):"—"}; })); }
  hit.addEventListener("pointermove",at); hit.addEventListener("pointerleave",()=>{cross.setAttribute("visibility","hidden");hideTip();});
  el.append(svg);
}

// ------------------------------------------------------- stacked columns
function stackedColumns(el, opt){
  const W=Math.max(280, el.clientWidth), H=opt.height||260, m={l:52,r:120,t:12,b:30};
  el.replaceChildren();
  if(!opt.rows.length){ el.append(h("div",{class:"empty"},"No timing data yet")); return; }
  const totals=opt.rows.map(r=>opt.cats.reduce((a,c)=>a+(r.values[c.name]||0),0));
  // Ticks on whole minutes once steps take minutes, so the axis reads 0 / 2 / 4 min, not 3.3 min.
  const unit=Math.max(...totals)>=120?60:1;
  const yT=niceTicks(0,Math.max(...totals)*1.05/unit,4).map(v=>v*unit), y1=yT[yT.length-1];
  const Y=v=>m.t+(1-v/y1)*(H-m.t-m.b);
  const band=(W-m.l-m.r)/opt.rows.length, bw=Math.min(24, band*0.7);
  const svg=s("svg",{viewBox:`0 0 ${W} ${H}`,height:H,role:"img","aria-label":opt.aria||""});
  for(const t of yT){ svg.append(s("line",{x1:m.l,x2:W-m.r,y1:Y(t),y2:Y(t),stroke:t===0?"var(--axis)":"var(--grid)","stroke-width":1})); svg.append(txt(m.l-6,Y(t)+4,fsec(t),{"text-anchor":"end"})); }
  const xStep=Math.max(1,Math.ceil(opt.rows.length/Math.max(2,Math.floor((W-m.l-m.r)/30))));
  opt.rows.forEach((r,i)=>{
    const cx=m.l+band*(i+0.5); if(i%xStep===0) svg.append(txt(cx,H-m.b+16,String(r.x),{"text-anchor":"middle"}));
    let acc=0; const present=opt.cats.filter(c=>(r.values[c.name]||0)>0);
    present.forEach((c,j)=>{ const v=r.values[c.name]; const yTop=Y(acc+v), yBot=Y(acc); acc+=v;
      const top=j===present.length-1, gap=top?0:2, hgt=Math.max(0,yBot-yTop-gap), x=cx-bw/2, y=yTop+gap;
      let d; if(top&&hgt>4){ const rr=4; d=`M${x},${y+hgt}V${y+rr}Q${x},${y} ${x+rr},${y}H${x+bw-rr}Q${x+bw},${y} ${x+bw},${y+rr}V${y+hgt}Z`; } else d=`M${x},${y}h${bw}v${hgt}h${-bw}Z`;
      const p=s("path",{d,fill:c.color}); svg.append(p);
      if(i===opt.rows.length-1 && yBot-yTop>=16){ const ly=(yTop+yBot)/2+4; svg.append(s("line",{x1:x+bw+3,x2:W-m.r+6,y1:ly-4,y2:ly-4,stroke:"var(--axis)","stroke-width":1})); svg.append(txt(W-m.r+10,ly,c.name,{fill:"var(--text-secondary)"})); }
    });
    const hit=s("rect",{x:cx-band/2,y:m.t,width:band,height:H-m.t-m.b,fill:"transparent",tabindex:0});
    const rows=[...opt.cats].reverse().filter(c=>r.values[c.name]!=null).map(c=>({name:c.name,color:c.color,box:true,value:fsec(r.values[c.name])}));
    rows.push({name:"Total",color:"transparent",box:true,value:fsec(totals[i])});
    hit.addEventListener("pointermove",ev=>showTip(ev,"Step "+r.x,rows)); hit.addEventListener("pointerleave",hideTip);
    hit.addEventListener("focus",ev=>{const b=hit.getBoundingClientRect(); showTip({clientX:b.right,clientY:b.top},"Step "+r.x,rows);}); hit.addEventListener("blur",hideTip);
    svg.append(hit);
  });
  el.append(svg);
}

// ------------------------------------------------------------- layout
function tile(label, value, cmp){ return h("div",{class:"tile"},h("div",{class:"label"},label),h("div",{class:"value"},value),h("div",{class:"cmp"},cmp)); }
function card(title, desc, ...kids){ return h("section",{class:"card"},h("h2",{},title),desc?h("p",{class:"desc"},desc):null,...kids); }
function table(cols, rows){ const t=h("table",{}); t.append(h("thead",{},h("tr",{},...cols.map(c=>h("th",{},c))))); const b=h("tbody",{}); for(const r of rows) b.append(h("tr",{},...r.map(v=>h("td",{},v)))); t.append(b); return h("div",{class:"tablewrap"},t); }

const C1="var(--series-1)", C2="var(--series-2)";
const charts=[];
function render(){
  app.replaceChildren();
  if(D.demo) app.append(h("div",{class:"banner"},"Synthetic demo data — this checks that the report renders. It is not a training run."));
  const env=D.env||{};
  app.append(h("h1",{},"SkyRL CholecT50 GRPO on Cloud TPU"));
  app.append(h("p",{class:"sub"},`Run ${D.run_id} · ${[env.accelerator, env.skyrl&&("SkyRL "+env.skyrl), env.engine&&("rollout engine "+env.engine)].filter(Boolean).join(" · ")||"environment not recorded"} · generated ${D.generated}`));

  const T=D.tpu, G=D.gpu;
  const cmp=(t,g)=> t==null ? `GPU: ${f3(g)}` : g==null ? "GPU: not reached" : `GPU: ${f3(g)} · ${t>=g?"+":"−"}${f3(Math.abs(t-g))} vs GPU`;
  app.append(h("div",{class:"tiles"},
    tile("Val reward before training", f3(T.eval_before), cmp(T.eval_before,G.eval_before)),
    tile("Val reward after training", f3(T.eval_after), G.eval_after==null?"GPU: not reached (final save OOM)":cmp(T.eval_after,G.eval_after)),
    tile("Training steps", `${T.steps} / ${T.expected_steps}`, `GPU: ${G.steps} / 16 before its OOM`),
    tile("Mean step time", fsec(T.mean_step_s), `Wall time ${fsec(T.wall_time_s)} · GPU ${fsec(G.wall_time_s)}`)));

  const series=[{name:G.label,short:"GPU",color:C1,points:G.reward},{name:T.label,short:"TPU",color:C2,points:T.reward}];
  const rewardEl=h("div",{class:"chart"});
  app.append(card("Training reward per step","Mean blended triplet-F1 reward over the 128 rollouts of each step (reward/avg_raw_reward). Same data, prompts, reward and hyperparameters on both.",
    legend(series), rewardEl,
    h("details",{},h("summary",{},"Table"),table(["Step","GPU","TPU"], Array.from({length:Math.max(16,T.reward.length)},(_,i)=>{const st=i+1, g=G.reward.find(p=>p[0]===st), t=T.reward.find(p=>p[0]===st); return [st,g?f3(g[1]):"—",t?f3(t[1]):"—"];})))));
  charts.push(()=>lineChart(rewardEl,{series,endLabels:true,yMin:0,xMin:1,xMax:16,xLabel:"step",aria:"Training reward per step, TPU vs GPU"}));

  const cats=D.timing.categories.map((n,i)=>({name:n,color:`var(--series-${i+1})`}));
  const timeEl=h("div",{class:"chart"});
  app.append(card("Where each TPU step's time goes","Seconds per training step by phase (SkyRL timing/*). Weight sync = full HF export on the trainer + rollout server restart on the new weights.",
    legend(cats,true), timeEl,
    h("details",{},h("summary",{},"Table"),table(["Step",...cats.map(c=>c.name),"Total"], D.timing.rows.map(r=>[r.x,...cats.map(c=>fsec(r.values[c.name])),fsec(Object.values(r.values).reduce((a,b)=>a+b,0))])))));
  charts.push(()=>stackedColumns(timeEl,{rows:D.timing.rows,cats,aria:"Seconds per step by phase"}));

  if(D.diagnostics.length){
    const grid=h("div",{class:"multiples"});
    for(const dg of D.diagnostics){ const el=h("div",{class:"chart"}); grid.append(card(dg.title,dg.key,el)); charts.push(()=>lineChart(el,{series:[{name:dg.title,color:C2,points:dg.points}],height:150,unit:dg.unit,xMin:1,xMax:16,aria:dg.title})); }
    app.append(h("h2",{style:"margin:24px 0 8px"},"TPU training diagnostics"), grid);
  }
  for(const c of charts) c();
}
render();
let rt; window.addEventListener("resize",()=>{clearTimeout(rt); rt=setTimeout(()=>{for(const c of charts) c();},120);});
</script></body></html>
"""


def write_report(data, out_html):
    payload = json.dumps(data).replace("</", "<\\/")
    with open(out_html, "w") as f:
        f.write(PAGE.replace("__DATA__", payload))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="logs/<run_id> (contains metrics.jsonl)")
    ap.add_argument("--gpu-ref", default=GPU_REF)
    ap.add_argument("--demo", action="store_true", help="write synthetic metrics into run_dir first (rendering check)")
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    metrics = os.path.join(args.run_dir, "metrics.jsonl")
    if args.demo:
        write_demo_metrics(metrics)
    if not os.path.exists(metrics):
        sys.exit(f"{metrics} not found (is SKYRL_METRICS_JSONL set by run_cholect50_tpu.sh?)")
    data = build(args.run_dir, metrics, args.gpu_ref, demo=args.demo)
    out_html = os.path.join(args.run_dir, "report.html")
    write_report(data, out_html)
    with open(os.path.join(args.run_dir, "final_results.json"), "w") as f:
        json.dump(final_results(data), f, indent=2)
    print(f"wrote {out_html}")
    print(f"wrote {os.path.join(args.run_dir, 'final_results.json')}")


if __name__ == "__main__":
    main()

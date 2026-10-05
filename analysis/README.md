# Analysis

Shared code and the canonical metrics schema. All comparison of the three
platforms happens here.

- [`schema.md`](schema.md) — the exact JSON schema every platform writes its
  `final_results.json` in.
- [`compare_platforms.py`](compare_platforms.py) — the cross-platform script.
  Auto-discovers every `*/metrics/final_results.json`, joins them on the
  `experiment` key, and writes side-by-side charts + a metric table.
- [`plot_results.py`](plot_results.py) — the original Fireworks-only plotting
  script (kept for the raw Fireworks `*_metrics.jsonl` loss curves). Superseded
  by `compare_platforms.py` for cross-platform comparison.
- [`plots/`](plots/) — generated PNG charts.

## How comparison works

Each platform writes:

```
<platform>/metrics/final_results.json
```

`compare_platforms.py` reads all of them and joins on the `experiment` key. That
is why the schema must match exactly — the join is on identical field names.
Runs that are `BLOCKED` / `NOT_STARTED` (or that simply omit a curve field) are
skipped gracefully rather than erroring, and are flagged "missing" on the chart.

### Running it

```bash
# needs matplotlib (e.g. `pip install matplotlib`, or use a venv)
python3 analysis/compare_platforms.py
```

Outputs (under `analysis/plots/`):

| File | What it shows |
|---|---|
| `platform_comparison_summary.png` | final metric per experiment, one bar per platform |
| `sft_loss_curves.png` | SFT train-loss curves (platforms that log them: Fireworks JSONL, Prime Intellect CSV) |
| `rl_curves.png` | RL reward/score curves (platforms that log them) |

It also prints a markdown table of each experiment's final metric by platform to
stdout.

## Fields to compare

For every experiment, side by side:

1. **Performance** — final metric (train loss, exact_match, micro_f1).
2. **Cost** — `estimated_cost_usd` (and per-token / per-GPU-hour where the
   platform exposes it).
3. **Time** — `wall_time_s`.
4. **Feature gaps** — anything the platform could not do, recorded as a finding.

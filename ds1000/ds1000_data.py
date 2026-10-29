#!/usr/bin/env python3
"""Build a verl-format parquet for DS-1000 RLVR.

Loads `xlangai/DS-1000` (1,000 problems, single 'test' split), keeps the code
completion prompt, and writes train.parquet/val.parquet with the columns verl
expects (`data_source`, `prompt`, `ability`, `reward_model`). The reward is the
simplified pass/fail `ds1000_reward.py`.

Usage:
    python ds1000_data.py --out_dir ~/cholec/ds1000_parquet [--library pandas] [--max_problems 160]
"""

import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def build(args):
    raw = load_dataset("xlangai/DS-1000")["test"]
    if args.library:
        raw = raw.filter(lambda r: r["metadata"]["library"].lower() == args.library.lower())
    if args.max_problems:
        raw = raw.select(range(min(args.max_problems, len(raw))))

    rows = []
    for r in raw:
        rows.append({
            "data_source": "ds1000",
            "prompt": r["prompt"],  # the code-completion context + problem
            "ability": "code",
            "reward_model": {"ground_truth": r.get("reference_code", "")},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="~/cholec/ds1000_parquet")
    ap.add_argument("--library", default=None, help="e.g. pandas; omit for all 7 libraries")
    ap.add_argument("--max_problems", type=int, default=None)
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    import os

    out = os.path.expanduser(args.out_dir)
    os.makedirs(out, exist_ok=True)

    rows = build(args)
    n_val = max(1, int(len(rows) * args.val_frac))
    train, val = rows[:-n_val], rows[-n_val:]

    pq.write_table(pa.Table.from_pylist(train), f"{out}/train.parquet")
    pq.write_table(pa.Table.from_pylist(val), f"{out}/val.parquet")
    print(f"wrote {len(train)} train / {len(val)} val to {out}")


if __name__ == "__main__":
    main()

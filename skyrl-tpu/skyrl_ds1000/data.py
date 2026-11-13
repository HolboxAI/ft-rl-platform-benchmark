"""Split DS-1000 into train / val and convert it to SkyRL parquet.

DS-1000 (xlangai/DS-1000, CC-BY-SA-4.0) has one split, ``test``, of 1000 problems: 452 originals and 548
rewrites of them. ``metadata.perturbation_origin_id`` names the original by its ``library_problem_id``, which is
numbered per library, so a group is (library, perturbation_origin_id). The split is made per group, so a problem
and its rewrites are always on the same side; otherwise val would measure memorised train answers. Within each
library, groups are taken in a ``--seed`` order until val holds ``--val_frac`` of that library's problems.

SkyRL's dataset contract:

    prompt       -> [{"role": "user", "content": <DS-1000 prompt + INSTRUCTION>}]
    env_class    -> "ds1000"
    reward_spec  -> {"method": "rule", "ground_truth": <json: problem_id, library, code_context>}
    extra_info   -> problem_id, library, perturbation_type, perturbation_origin_id

    python data.py --src ds1000.jsonl.gz --out_dir ~/ds1000/skyrl_parquet      # DS-1000 repo's data file
    python data.py --src test.parquet    --out_dir ~/ds1000/skyrl_parquet      # Hugging Face parquet
"""
import argparse
import gzip
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd

ENV_CLASS = "ds1000"

# DS-1000's prompts are written for completion models and stop at "BEGIN SOLUTION\n<code>". A chat model is told
# what to return; ds1000_env.extract_code reads a fenced block or a <code> block.
INSTRUCTION = (
    "\n\nWrite only the code that goes after BEGIN SOLUTION, so that the program above runs and sets the "
    "requested variable(s). Do not repeat the setup code. Reply with the code inside a single ```python block."
)


def load_problems(src: str) -> List[Dict]:
    if src.endswith(".jsonl.gz"):
        with gzip.open(src, "rt") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif src.endswith(".jsonl"):
        with open(src) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        rows = pd.read_parquet(src).to_dict("records")
    for r in rows:
        if isinstance(r["metadata"], str):
            r["metadata"] = json.loads(r["metadata"])
    rows.sort(key=lambda r: int(r["metadata"]["problem_id"]))
    if len(rows) != 1000:
        raise SystemExit(f"expected 1000 DS-1000 problems, got {len(rows)}")
    return rows


def group_key(r: Dict) -> Tuple[str, int]:
    md = r["metadata"]
    return md["library"], int(md["perturbation_origin_id"])


def split_groups(rows: List[Dict], val_frac: float, seed: int) -> set:
    """Return the (library, perturbation_origin_id) groups that go to val."""
    groups_by_lib = defaultdict(Counter)
    for r in rows:
        lib, origin = group_key(r)
        groups_by_lib[lib][origin] += 1
    rng = random.Random(seed)
    val = set()
    for lib in sorted(groups_by_lib):
        sizes = groups_by_lib[lib]
        target = max(1, round(sum(sizes.values()) * val_frac))
        origins = sorted(sizes)
        rng.shuffle(origins)
        taken = 0
        for origin in origins:
            if taken >= target:
                break
            val.add((lib, origin))
            taken += sizes[origin]
    return val


def convert_row(r: Dict) -> Dict:
    md = r["metadata"]
    gt = {"problem_id": int(md["problem_id"]), "library": md["library"], "code_context": r["code_context"]}
    return {
        "prompt": [{"role": "user", "content": r["prompt"].rstrip() + INSTRUCTION}],
        "env_class": ENV_CLASS,
        "reward_spec": {"method": "rule", "ground_truth": json.dumps(gt)},
        "extra_info": {
            "problem_id": int(md["problem_id"]),
            "library": md["library"],
            "perturbation_type": md["perturbation_type"],
            "perturbation_origin_id": int(md["perturbation_origin_id"]),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DS-1000 ds1000.jsonl.gz (GitHub) or test.parquet (Hugging Face)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_problems(args.src)
    val_groups = split_groups(rows, args.val_frac, args.seed)
    splits = {"train": [], "val": []}
    for r in rows:
        side = "val" if group_key(r) in val_groups else "train"
        splits[side].append(r)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.src, "rb") as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {"src": os.path.basename(args.src), "src_sha256": src_sha, "val_frac": args.val_frac, "seed": args.seed}
    for side, items in splits.items():
        out = os.path.join(args.out_dir, f"{side}.parquet")
        pd.DataFrame([convert_row(r) for r in items]).to_parquet(out)
        by_lib = defaultdict(int)
        for r in items:
            by_lib[r["metadata"]["library"]] += 1
        manifest[side] = {
            "rows": len(items),
            "by_library": dict(sorted(by_lib.items())),
            "problem_ids": [int(r["metadata"]["problem_id"]) for r in items],
        }
        print(f"wrote {out} ({len(items)} rows) {dict(sorted(by_lib.items()))}", flush=True)
    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

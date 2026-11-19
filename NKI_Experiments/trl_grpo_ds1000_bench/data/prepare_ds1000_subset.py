"""Pick a fixed 32-problem subset of DS-1000 and save it as JSONL.

Run this ONCE (on any machine that has `transformers`), commit/copy the output,
and use the same file on every platform so all runs train on identical prompts.

Selection rules (deterministic for a given seed):
  * drop libraries listed in config["exclude_libraries"] (slow / fragile to execute)
  * keep only problems whose chat-formatted prompt fits in max_prompt_length tokens
  * sample round-robin across the remaining libraries so the mix stays balanced

Usage:
  python -m data.prepare_ds1000_subset --config configs/grpo_ds1000.json \
      --source ../data/ds1000/test.jsonl
"""

import argparse
import json
import os
import random
from collections import defaultdict

SYSTEM_PROMPT = (
    "You are an expert Python data-science programmer. Complete the solution. "
    "Reply with only the code that goes between BEGIN SOLUTION and END SOLUTION, "
    "inside a single ```python code block. Do not repeat the setup code."
)


def build_messages(ds1000_prompt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ds1000_prompt},
    ]


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(here, "configs/grpo_ds1000.json"))
    p.add_argument("--source", default=os.path.join(here, "../data/ds1000/test.jsonl"))
    p.add_argument("--out", default=None, help="defaults to config['subset_file']")
    p.add_argument("--no-tokenizer", action="store_true",
                   help="skip the token-length filter (uses a 4-chars/token estimate instead)")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    out = args.out or os.path.join(here, cfg["subset_file"])
    n, seed, max_len = cfg["num_instances"], cfg["subset_seed"], cfg["max_prompt_length"]

    rows = [json.loads(l) for l in open(args.source)]
    rows = [r for r in rows if r["metadata"]["library"] not in set(cfg["exclude_libraries"])]

    if args.no_tokenizer:
        def n_tokens(r):
            return sum(len(m["content"]) for m in build_messages(r["prompt"])) // 4 + 30
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model_name"])

        def n_tokens(r):
            return len(tok.apply_chat_template(build_messages(r["prompt"]),
                                               add_generation_prompt=True, tokenize=True))

    by_lib = defaultdict(list)
    for r in rows:
        r["_n_tokens"] = n_tokens(r)
        if r["_n_tokens"] <= max_len:
            by_lib[r["metadata"]["library"]].append(r)

    rng = random.Random(seed)
    for lib in by_lib:
        rng.shuffle(by_lib[lib])
    libs = sorted(by_lib)
    picked = []
    while len(picked) < n and any(by_lib.values()):
        for lib in libs:
            if by_lib[lib] and len(picked) < n:
                picked.append(by_lib[lib].pop())
    if len(picked) < n:
        raise SystemExit(f"only {len(picked)} problems fit in {max_len} tokens; raise max_prompt_length")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in picked:
            md = r["metadata"]
            f.write(json.dumps({
                "problem_id": md["problem_id"],
                "library": md["library"],
                "prompt_tokens": r["_n_tokens"],
                "prompt": build_messages(r["prompt"]),
                "code_context": r["code_context"],
                "reference_code": r["reference_code"],
            }) + "\n")

    counts = defaultdict(int)
    for r in picked:
        counts[r["metadata"]["library"]] += 1
    print(f"wrote {len(picked)} problems -> {out}")
    print("per library:", dict(counts))
    print("max prompt tokens:", max(r["_n_tokens"] for r in picked))


if __name__ == "__main__":
    main()

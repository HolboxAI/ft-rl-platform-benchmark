"""Check the DS-1000 split and reward on CPU, before any TPU time is spent.

    python check_reward.py --ds1000_repo ~/DS-1000 --out_dir ~/ds1000/check

``--ds1000_repo`` is a clone of github.com/xlang-ai/DS-1000 (its ``data/ds1000.jsonl.gz``, its published model
answers and their scores in ``results/``). ``DS1000_PYTHON`` must point at the venv from ``setup_exec_venv.sh``.

Checks (each prints PASS / FAIL; exit code 1 if any fails):

1. split      - data.py's train / val: 1000 rows in all, no (library, original) group shared by train and val, every library in val
2. reference  - every ``reference_code`` passes its tests (the reward can reach 1.0 on every problem)
3. empty      - an empty solution and ``pass`` pass no problem (the reward is not free)
4. hacks      - ``os._exit(0)``, ``sys.exit(0)`` and an endless loop fail on a sample of problems
5. official   - DS-1000's published answers of two models, through DS-1000's own postprocess, score what
                DS-1000's ``results/`` reports for them, within 0.01 (the reward matches the official scorer)
6. env        - ``DS1000Env.step`` on chat-style replies, all 1000 problems: reference code in a fenced block
                with prose = 1.0, no code = 0.0
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sandbox  # noqa: E402

RESULTS = {}


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS[name] = {"ok": ok, "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def run_all(problems, solutions, timeout, workers):
    def one(i):
        return sandbox.run_test(problems[i]["code_context"], solutions[i], timeout=timeout)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, range(len(problems))))
    return out, time.time() - t0


def official_postprocess(code):
    """DS-1000's test_ds1000.postprocess, verbatim."""
    if type(code) is list:
        code = code[0]
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.split("\nEND SOLUTION")[0]
    code = code.replace("<code>", "")
    return code


def check_split(src, out_dir):
    pq = os.path.join(out_dir, "skyrl_parquet")
    subprocess.run([sys.executable, os.path.join(HERE, "data.py"), "--src", src, "--out_dir", pq], check=True)
    tr, va = (pd.read_parquet(os.path.join(pq, f"{s}.parquet")) for s in ("train", "val"))
    o_tr = {(e["library"], e["perturbation_origin_id"]) for e in tr.extra_info}
    o_va = {(e["library"], e["perturbation_origin_id"]) for e in va.extra_info}
    libs_va = Counter(e["library"] for e in va.extra_info)
    ok = len(tr) + len(va) == 1000 and not (o_tr & o_va) and len(libs_va) == 7 and set(tr.env_class) == {"ds1000"}
    report("split", ok, f"train {len(tr)}, val {len(va)}, shared originals {len(o_tr & o_va)}, val by library {dict(libs_va)}")
    return tr, va


def check_env(problems, workers):
    try:
        import skyrl_gym  # noqa: F401
    except ImportError:  # the checker venv has no SkyRL; give the env the base class it needs
        import types

        class _Out(dict):
            pass

        class _Base:
            def __init__(self):
                self.turns = 0

        mod = types.ModuleType("skyrl_gym.envs.base_text_env")
        mod.BaseTextEnv, mod.BaseTextEnvStepOutput = _Base, lambda **kw: _Out(**kw)
        for name in ("skyrl_gym", "skyrl_gym.envs"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["skyrl_gym.envs.base_text_env"] = mod
    from ds1000_env import DS1000Env

    def step(p, reply):
        gt = json.dumps({"problem_id": p["metadata"]["problem_id"], "library": p["metadata"]["library"],
                         "code_context": p["code_context"]})
        return DS1000Env(extras={"reward_spec": {"method": "rule", "ground_truth": gt}}).step(reply)["reward"]

    def fenced(p):
        return step(p, f"Here is the solution:\n\n```python\n{p['reference_code']}\n```\n\nThis sets the variable.")

    with ThreadPoolExecutor(workers) as ex:
        rewards_ok = list(ex.map(fenced, problems))
    rewards_none = [step(p, "I am not sure how to do this.") for p in problems]
    failed = [p["metadata"]["problem_id"] for p, r in zip(problems, rewards_ok) if r != 1.0]
    ok = not failed and all(r == 0.0 for r in rewards_none)
    report("env", ok, f"{len(problems)} problems: fenced reference -> {sum(rewards_ok):.0f}/{len(problems)} reward 1.0 "
                      f"(failing: {failed[:20]}); no code -> {sum(rewards_none)} total reward")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds1000_repo", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default="gpt-4-turbo-2024-04-09,codex002",
                    help="models whose published answers and scores are in the DS-1000 repo (check 5)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=float, default=sandbox.DEFAULT_TIMEOUT_S)
    ap.add_argument("--checks", default="split,reference,empty,hacks,official,env", help="comma-separated subset")
    args = ap.parse_args()
    checks = set(args.checks.split(","))
    os.makedirs(args.out_dir, exist_ok=True)
    src = os.path.join(args.ds1000_repo, "data", "ds1000.jsonl.gz")
    with gzip.open(src, "rt") as f:
        problems = sorted((json.loads(line) for line in f), key=lambda p: int(p["metadata"]["problem_id"]))
    print(f"DS1000_PYTHON={sandbox.exec_python()}  network isolated: {sandbox.net_isolated()}  workers: {args.workers}")

    if "split" in checks:
        check_split(src, args.out_dir)

    if "reference" in checks:
        ref, dt = run_all(problems, [p["reference_code"] for p in problems], args.timeout, args.workers)
        failed = [(i, r["status"], r["error"][-300:]) for i, r in enumerate(ref) if not r["passed"]]
        by_lib = Counter(problems[i]["metadata"]["library"] for i, _, _ in failed)
        report("reference", not failed, f"{1000 - len(failed)}/1000 pass in {dt:.0f}s; failing by library {dict(by_lib)}")
        with open(os.path.join(args.out_dir, "reference_failures.json"), "w") as f:
            json.dump(failed, f, indent=1)

    for name, sol in (("empty", ""), ("pass", "pass")) if "empty" in checks else ():
        res, dt = run_all(problems, [sol] * 1000, args.timeout, args.workers)
        n = sum(r["passed"] for r in res)
        report(f"empty[{name}]", n == 0, f"{n}/1000 pass ({dt:.0f}s)")

    sample = problems[::25]
    hacks = {"os._exit": "import os\nos._exit(0)", "sys.exit": "import sys\nsys.exit(0)", "loop": "while True:\n    pass"}
    for name, sol in hacks.items() if "hacks" in checks else ():
        res, _ = run_all(sample, [sol] * len(sample), 10.0, args.workers)
        statuses = Counter(r["status"] for r in res)
        report(f"hacks[{name}]", not any(r["passed"] for r in res), f"{len(sample)} problems -> {dict(statuses)}")

    for model in args.models.split(",") if "official" in checks else ():
        with open(os.path.join(args.ds1000_repo, "data", f"{model}-answers.jsonl")) as f:
            answers = [json.loads(line) for line in f]
        # the answer files name the problem by "id" or, for codex002, by metadata.problem_id
        answers = sorted(answers, key=lambda a: int(a["id"] if "id" in a else a["metadata"]["problem_id"]))
        answers = [official_postprocess(a["code"]) for a in answers]
        res, dt = run_all(problems, answers, args.timeout, args.workers)
        ours = sum(r["passed"] for r in res) / 1000
        per_lib = defaultdict(list)
        for p, r in zip(problems, res):
            per_lib[p["metadata"]["library"]].append(r["passed"])
        with open(os.path.join(args.ds1000_repo, "results", f"{model}-result.txt")) as f:
            m = re.search(r"mean\s+([0-9.]+)", f.read())
        published = float(m.group(1)) if m else None
        ok = published is not None and abs(ours - published) <= 0.01
        report(f"official[{model}]", ok, f"ours {ours:.3f}, published {published} ({dt:.0f}s); "
                                         f"by library {({k: round(sum(v) / len(v), 3) for k, v in sorted(per_lib.items())})}")
        with open(os.path.join(args.out_dir, f"{model}-rescored.json"), "w") as f:
            json.dump([{"id": i, **r} for i, r in enumerate(res)], f)

    if "env" in checks:
        check_env(problems, args.workers)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"net_isolated": sandbox.net_isolated(), "checks": RESULTS}, f, indent=1)
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    print("ALL PASS" if not bad else f"FAILED: {bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

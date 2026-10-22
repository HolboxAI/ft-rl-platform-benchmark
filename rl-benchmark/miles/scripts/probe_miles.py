#!/usr/bin/env python3
"""On-hardware probe: how far does Miles get on this accelerator?

Miles is CUDA-first. It has an experimental TPU platform hook
(`MILES_ACCELERATOR=tpu` -> `miles/platforms/tpu.py`, "TPU v7x through
torch_tpu") and no Trainium/Neuron platform. This probe runs on the
accelerator host, imports Miles piece by piece, and records which stages pass:

  1. accelerator     the chip's PyTorch backend makes a tensor
  2. miles_import    `import miles` (installed with --no-deps; see run_probe.sh)
  3. platform        MILES_ACCELERATOR=<value> -> current_platform.current_device()
  4. train_backends  import fsdp_utils / torchtitan_utils / megatron_utils
  5. rollout         import sglang + miles.rollout.sglang_rollout (the GRPO rollout path)
  6. import_surface  import every miles module; group failures by the missing package

    python miles/scripts/probe_miles.py --backend tpu --miles-src ~/miles --out miles/logs/probe_tpu.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import re
import sys
import time
import traceback
from collections import Counter


def try_stage(res, name, fn):
    t = time.perf_counter()
    try:
        extra = fn() or {}
        res["stages"][name] = {"ok": True, "s": round(time.perf_counter() - t, 2), **extra}
        print(f"[probe] {name}: ok {extra or ''}", flush=True)
        return True
    except BaseException as e:  # SystemExit/ImportError from deep inside Miles count too
        res["stages"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1500],
                               "traceback": traceback.format_exc()[-2500:]}
        print(f"[probe] {name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("tpu", "neuron", "cuda"), required=True)
    ap.add_argument("--miles-src", default=os.path.expanduser("~/miles"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.path.insert(0, args.miles_src)

    # Miles only knows "cuda" and "tpu". For Trainium we pass "neuron" on purpose:
    # the expected ValueError at stage 3 is the finding.
    os.environ["MILES_ACCELERATOR"] = {"tpu": "tpu", "neuron": "neuron", "cuda": "cuda"}[args.backend]
    res = {"backend": args.backend, "MILES_ACCELERATOR": os.environ["MILES_ACCELERATOR"], "stages": {}}

    def accelerator():
        import torch

        if args.backend == "tpu":
            import torch_tpu  # noqa: F401
            dev = torch.device("tpu")
        elif args.backend == "neuron":
            import torch_neuronx  # noqa: F401
            import torch_xla.core.xla_model as xm
            dev = xm.xla_device()
        else:
            dev = torch.device("cuda")
        x = (torch.ones(4, device=dev) * 2).sum()
        return {"torch": torch.__version__, "device": str(dev), "check": float(x.cpu())}

    def miles_import():
        import miles

        return {"miles_path": os.path.dirname(miles.__file__)}

    def platform():
        from miles.platforms import current_platform

        return {"platform": type(current_platform).__name__,
                "device": str(current_platform.current_device())}

    def train_backends():
        out = {}
        for mod in ("miles.backends.fsdp_utils", "miles.backends.torchtitan_utils", "miles.backends.megatron_utils"):
            try:
                importlib.import_module(mod)
                out[mod] = "ok"
            except BaseException as e:
                out[mod] = f"{type(e).__name__}: {str(e)[:200]}"
        if all(v != "ok" for v in out.values()):
            raise ImportError(f"no training backend importable: {out}")
        return {"backends": out}

    def rollout():
        import sglang  # noqa: F401
        importlib.import_module("miles.rollout.sglang_rollout")
        return {"sglang": getattr(sys.modules["sglang"], "__version__", "?")}

    def import_surface():
        import miles

        ok, failed, missing = 0, 0, Counter()

        def record(e):
            hit = re.search(r"No module named '([^'.]+)", str(e))
            missing[hit.group(1) if hit else type(e).__name__] += 1

        def on_pkg_error(name):  # walk_packages imports packages to recurse into them
            nonlocal failed
            failed += 1
            record(sys.exc_info()[1])

        for m in pkgutil.walk_packages(miles.__path__, "miles.", onerror=on_pkg_error):
            try:
                importlib.import_module(m.name)
                ok += 1
            except BaseException as e:
                failed += 1
                record(e)
        return {"modules_ok": ok, "modules_failed": failed, "failures_by_cause": dict(missing.most_common(15))}

    try_stage(res, "accelerator", accelerator)
    if try_stage(res, "miles_import", miles_import):
        try_stage(res, "platform", platform)
        try_stage(res, "train_backends", train_backends)
        try_stage(res, "rollout", rollout)
        try_stage(res, "import_surface", import_surface)
    res["ok"] = all(s["ok"] for s in res["stages"].values())
    res["first_failure"] = next((k for k, s in res["stages"].items() if not s["ok"]), None)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    print(json.dumps({k: (v if k != "stages" else {n: s["ok"] for n, s in v.items()}) for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Static porting audit: how much of an RL framework is tied to NVIDIA?

Scans a checked-out source tree (it never imports or runs it) and counts the
call sites and imports that assume CUDA, so the porting effort to Trainium or
TPU can be stated as numbers rather than adjectives.

    git clone --depth 1 https://github.com/radixark/miles /tmp/src/miles
    python -m common.porting_audit /tmp/src/miles --name miles --out miles/metrics/porting_audit.json

Categories (regexes over .py source, comments stripped):

  cuda_runtime    torch.cuda.*, .cuda(), device="cuda", torch.device("cuda")
  nccl            NCCL backends, pynccl, cupy, NVSHMEM
  gpu_kernels     triton, flash_attn, flashinfer, deep_ep, deep_gemm, apex, transformer_engine, causal_conv1d, fla
  nvidia_stack    megatron, nvidia.*, pynvml / nvidia-smi
  sglang / vllm   rollout engines (CUDA builds; TPU variants are separate projects)
  jax             jax / flax (a hardware-portable path, counted as a positive signal)
  xla_or_neuron   torch_xla, torch_neuronx, torch_tpu, libneuronxla, jax-neuronx (existing non-NVIDIA support)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict

CATEGORIES = {
    "cuda_runtime": [r"\btorch\.cuda\.", r"\.cuda\(\)", r"""device\s*=\s*["']cuda""", r"""torch\.device\(\s*["']cuda""",
                     r"\bcuda_visible_devices\b", r"\bCUDA_VISIBLE_DEVICES\b"],
    "nccl": [r"\bnccl\b", r"\bpynccl\b", r"\bcupy\b", r"\bnvshmem\b"],
    "gpu_kernels": [r"^\s*(import|from)\s+(triton|flash_attn|flashinfer|deep_ep|deep_gemm|apex|transformer_engine|"
                    r"causal_conv1d|fla|liger_kernel|sgl_kernel)\b"],
    "nvidia_stack": [r"^\s*(import|from)\s+(megatron|nvidia|pynvml)\b", r"nvidia-smi"],
    "sglang": [r"^\s*(import|from)\s+sglang\b"],
    "vllm": [r"^\s*(import|from)\s+vllm\b"],
    "jax": [r"^\s*(import|from)\s+(jax|flax|optax)\b"],
    "xla_or_neuron": [r"^\s*(import|from)\s+(torch_xla|torch_neuronx|torch_tpu|libneuronxla|jax_neuronx)\b"],
}
COMPILED = {k: [re.compile(p, re.M | re.I if k in ("nccl",) else re.M) for p in v] for k, v in CATEGORIES.items()}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist"}
# Tests, docs and examples are not what a port has to change; --include-tests adds them back.
NON_PRODUCTION_DIRS = {"tests", "test", "docs", "examples", "benchmarks", "ci"}


def strip_comments(src: str) -> str:
    return "\n".join(line.split("#", 1)[0] if not line.lstrip().startswith(("'", '"')) else line
                     for line in src.splitlines())


def top_area(rel: str) -> str:
    parts = rel.split(os.sep)
    return os.sep.join(parts[:2]) if len(parts) > 2 else parts[0]


def audit(root: str, include_tests: bool = False) -> dict:
    hits = defaultdict(Counter)        # category -> file -> count
    areas = defaultdict(Counter)       # area -> category -> count
    n_files = n_lines = 0
    for dp, dns, fns in os.walk(root):
        skip = SKIP_DIRS if include_tests else SKIP_DIRS | NON_PRODUCTION_DIRS
        dns[:] = [d for d in dns if d not in skip]
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dp, fn)
            rel = os.path.relpath(path, root)
            try:
                src = strip_comments(open(path, encoding="utf-8", errors="ignore").read())
            except OSError:
                continue
            n_files += 1
            n_lines += src.count("\n") + 1
            for cat, pats in COMPILED.items():
                c = sum(len(p.findall(src)) for p in pats)
                if c:
                    hits[cat][rel] += c
                    areas[top_area(rel)][cat] += c
    commit = None
    try:
        commit = subprocess.run(["git", "-C", root, "log", "-1", "--format=%h %cs"],
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        pass
    return {
        "root": os.path.abspath(root),
        "commit": commit,
        "scope": "all .py" if include_tests else "production code (tests/docs/examples/benchmarks/ci excluded)",
        "py_files": n_files,
        "py_lines": n_lines,
        "totals": {cat: {"call_sites": sum(f.values()), "files": len(f)} for cat, f in
                   ((c, hits.get(c, Counter())) for c in CATEGORIES)},
        "top_files": {cat: f.most_common(8) for cat, f in hits.items()},
        "by_area": {a: dict(c) for a, c in sorted(areas.items(), key=lambda kv: -sum(kv[1].values()))[:15]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-tests", action="store_true")
    args = ap.parse_args()
    res = {"name": args.name or os.path.basename(os.path.abspath(args.root)), **audit(args.root, args.include_tests)}
    print(f"{res['name']} @ {res['commit']}: {res['py_files']} .py files, {res['py_lines']} lines")
    print(f"{'category':<15} {'call sites':>10} {'files':>6}")
    for cat, t in res["totals"].items():
        print(f"{cat:<15} {t['call_sites']:>10} {t['files']:>6}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

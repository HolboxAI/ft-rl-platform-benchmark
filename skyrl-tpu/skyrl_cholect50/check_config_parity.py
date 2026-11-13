#!/usr/bin/env python3
"""Check the TPU launch config against the GPU run's documented config.

Source of truth: ``ft-rl-platform-benchmark-git/skyrl/config/SKYRL_CONFIG.md``
(copied to ``config/gpu_reference_SKYRL_CONFIG.md``). Every ``| `key` | `value` |``
row in it is compared with the override the TPU script ``run_cholect50_tpu.sh``
passes (shell defaults resolved). The result is one of:

* ``same``       identical value;
* ``allowed``    differs, and the difference is listed in TPU_REQUIRED below with its reason;
* ``MISMATCH``   differs without a reason -> exit 1;
* ``MISSING``    the TPU script does not set it -> exit 1 (SkyRL defaults are not trusted
                 to match; every documented key must be passed explicitly).

    python skyrl_cholect50/check_config_parity.py            # prints the table, exit 0/1
    python skyrl_cholect50/check_config_parity.py --markdown # table for config/PARITY.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPU_MD = os.path.join(ROOT, "config", "gpu_reference_SKYRL_CONFIG.md")
TPU_SH = os.path.join(HERE, "run_cholect50_tpu.sh")

# Keys whose TPU value must differ, and why. Nothing else may differ.
TPU_REQUIRED = {
    "trainer.placement.colocate_all": (
        "false", "TorchTPU allows one process per chip, so the trainer and the rollout engine "
                 "cannot time-share chips the way FSDP + vLLM share a GPU with sleep/wake."),
    "generator.inference_engine.weight_sync_backend": (
        "reload", "nccl/CUDA-IPC need CUDA on both sides. reload = full HF export + rollout "
                  "server restart; same weights reach the sampler, slower."),
    "generator.inference_engine.run_engines_locally": (
        "false", "SkyRL can only launch CUDA vLLM itself; the TPU rollout server is external."),
    "generator.inference_engine.num_engines": (
        "1", "One external TPU rollout server instead of 4 local vLLM engines (throughput only; "
             "no effect on the algorithm). Set ROLLOUT_URLS to several servers for 4."),
    "trainer.policy.fsdp_config.cpu_offload": (
        "false", "Was needed to fit 24 GB A10Gs; 4 x 32 GB v6e chips hold params, grads and AdamW "
                 "state without it. Same math, no host round trip. CPU_OFFLOAD=true restores it."),
}
# Other accepted values for a key, by layout (value -> reason).
ALTERNATIVES = {
    "trainer.placement.ref_num_gpus_per_node": {
        "1": "TOPOLOGY=single: the ref (frozen, forward-only) runs on 1 chip so the rollout server fits on "
             "the same v6e-8 (2-chip TorchTPU groups fail to start; 1 and 4 work). Same model and math; "
             "the ref forward pass is not sharded.",
    },
}
# Keys not in SKYRL_CONFIG.md that the TPU script must set (and why).
TPU_EXTRA = {
    "trainer.placement.colocate_policy_ref": "false: implied by colocate_all=true on GPU; on TPU policy and ref need separate chips.",
    "generator.inference_engine.external_server_urls": "the TPU rollout server(s).",
    "trainer.flash_attn": "false: flash-attn is CUDA-only; sdpa attention instead.",
    "trainer.ref.fsdp_config.cpu_offload": "true only in TOPOLOGY=single: the ref runs on 1 chip, so its weights stay in host RAM "
                                          "and FSDP streams them per layer (same math). The GPU run offloaded all colocated models.",
}
# Values that are a host-specific path or count: compared by name only.
PATHS = {"data.train_data", "data.val_data", "trainer.policy.model.path", "trainer.export_path", "trainer.ckpt_path",
         "trainer.log_path"}
# Documented as the GPU count of the g5.12xlarge; on TPU the same number means TPU chips.
COUNTS = {"trainer.placement.policy_num_gpus_per_node", "trainer.placement.critic_num_gpus_per_node",
          "trainer.placement.ref_num_gpus_per_node"}


def parse_md(path):
    rows = {}
    for line in open(path):
        m = re.match(r"\|\s*`([a-z_][\w.]*)`\s*\|\s*`([^`]*)`[^|]*\|", line.strip())
        if m:
            rows[m.group(1)] = m.group(2).strip()
    # Hardware-only rows the GPU table expresses in prose:
    return rows


def parse_sh(path):
    text = open(path).read()
    defaults = dict(re.findall(r':\s*"\$\{(\w+):=([^}]*)\}"', text))
    defaults["#_urls[@]"] = "1"  # one ROLLOUT_URLS entry by default

    def resolve(v):
        v = v.strip().strip('"')
        v = re.sub(r"\$\{#_urls\[@\]\}", "1", v)
        v = re.sub(r"\$\{(\w+)\}|\$(\w+)", lambda m: defaults.get(m.group(1) or m.group(2), m.group(0)), v)
        v = re.sub(r"\$\(\((.+?)\)\)", lambda m: str(eval(m.group(1), {})) if re.fullmatch(r"[\d+\-* ]+", m.group(1)) else m.group(0), v)
        return v.strip("'\"")

    overrides = {}
    body = text.split('python "$HERE/entrypoint.py"', 1)[1]
    for m in re.finditer(r"^\s*([a-z_][\w.]*)=(\S+)", body, re.M):
        overrides[m.group(1)] = resolve(m.group(2))
    return overrides


def norm(v):
    v = v.strip().strip("'\"")
    return {"True": "true", "False": "false"}.get(v, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-md", default=GPU_MD)
    ap.add_argument("--tpu-sh", default=TPU_SH)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="value the launch actually uses (e.g. from the environment), overriding the script default")
    args = ap.parse_args()

    gpu = parse_md(args.gpu_md)
    tpu = parse_sh(args.tpu_sh)
    for kv in args.set:
        k, v = kv.split("=", 1)
        tpu[k] = v
    rows, bad = [], 0
    for key, gval in gpu.items():
        tval = tpu.get(key)
        if tval is None:
            status, note = "MISSING", "not passed by run_cholect50_tpu.sh"
            bad += 1
        elif key in PATHS:
            status, note = "same", "host path"
        elif norm(tval) == norm(gval):
            status, note = "same", ""
        elif key in TPU_REQUIRED and norm(tval) == TPU_REQUIRED[key][0]:
            status, note = "allowed", TPU_REQUIRED[key][1]
        elif norm(tval) in ALTERNATIVES.get(key, {}):
            status, note = "allowed", ALTERNATIVES[key][norm(tval)]
        else:
            status, note = "MISMATCH", ""
            bad += 1
        if key in COUNTS and status == "same":
            note = "4 GPUs -> 4 TPU chips"
        rows.append((key, gval, tval or "—", status, note))
    for key, why in TPU_EXTRA.items():
        status = "extra" if key in tpu else "MISSING"
        bad += status == "MISSING"
        rows.append((key, "(not in GPU config)", tpu.get(key, "—"), status, why))

    if args.markdown:
        print("| Parameter | GPU run | TPU run | Status | Note |\n|---|---|---|---|---|")
        for k, g, t, st, n in rows:
            print(f"| `{k}` | `{g}` | `{t}` | {st} | {n} |")
    else:
        w = max(len(r[0]) for r in rows)
        for k, g, t, st, n in rows:
            print(f"{st:8}  {k:<{w}}  gpu={g:<24} tpu={t:<24} {n[:90]}")
        counts = {s: sum(r[3] == s for r in rows) for s in ("same", "allowed", "extra", "MISMATCH", "MISSING")}
        print(f"\n{len(gpu)} documented GPU parameters: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
        print("PARITY OK" if bad == 0 else f"PARITY FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

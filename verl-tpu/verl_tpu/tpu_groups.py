#!/usr/bin/env python3
"""Print ``export SKYRL_TPU_GROUP_ENV_<GROUP>=...`` lines for a chip split.

TorchTPU runs one process per chip and needs its topology env set before the
backend starts. SkyRL builds a separate process group per actor group (policy,
ref), and on the trainer VM the two groups must use disjoint chips, because a
chip can be opened by one process only. The skyrl-tpu patch
(``skyrl/backends/skyrl_train/utils/device.py:configure_process_env``) reads,
per group::

    {"env": {...shared by the group's ranks...}, "per_rank": [{...rank 0...}, ...]}

    eval "$(python tpu_groups.py --group policy:0,1,2,3 --group ref:4,5,6,7)"

Two variable families are set, because it is not yet known which one TorchTPU 0.2
honours for chip subsets:

* TorchTPU's own (``TORCH_TPU_TOPOLOGY``, ``TORCH_TPU_SLICEBUILDER_ADDRESSES``,
  ``TPU_VISIBLE_DEVICES``), from the TorchTPU 0.2 docs;
* libtpu's per-process ones (``TPU_VISIBLE_CHIPS``, ``TPU_CHIPS_PER_PROCESS_BOUNDS``,
  ``TPU_PROCESS_BOUNDS``, ``TPU_PROCESS_ADDRESSES``, ``TPU_PROCESS_PORT``,
  ``CLOUD_TPU_TASK_ID``), the documented way to run several TPU processes on one
  host (one chip each); TorchTPU is built on libtpu/PJRT. ``--no-libtpu-env`` drops them.

UNVERIFIED: the docs do not say how to run two independent process groups on chip
subsets of one host. ``probes/p0_torchtpu.py`` tests exactly this layout; if it
fails, compare against ``python -m torch.tpu.distributed`` output on the VM and
pass ``--extra KEY=VALUE`` / ``--per-rank-key`` to match, or give each group its
own VM (README, "Topology B").
"""
import argparse
import json
import shlex

# chips -> (x, y, z) slice shape for a v6e host (2x4 chips; chip id = x + 2*y, so chips 4,5 are a 2x1 pair).
TOPOLOGY = {1: "1,1,1", 2: "2,1,1", 4: "2,2,1", 8: "2,4,1"}


def group_spec(chips, port_base, extra, per_rank_key, libtpu=True):
    n = len(chips)
    if n not in TOPOLOGY:
        raise SystemExit(f"unsupported group size {n}; use 1, 2, 4 or 8 chips")
    env = {
        "TORCH_TPU_TOPOLOGY": TOPOLOGY[n],
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{port_base + i}" for i in range(n)),
        "TPU_VISIBLE_DEVICES": ",".join(str(c) for c in chips),
    }
    proc_ports = [port_base + 8 + i for i in range(n)]
    if libtpu:
        env.update({
            "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
            "TPU_PROCESS_BOUNDS": TOPOLOGY[n],
            "TPU_PROCESS_ADDRESSES": ",".join(f"localhost:{p}" for p in proc_ports),
        })
    env.update(extra)
    per_rank = []
    for i, c in enumerate(chips):
        r = {per_rank_key: str(c)}
        if libtpu:
            r.update({"TPU_VISIBLE_CHIPS": str(c), "TPU_PROCESS_PORT": str(proc_ports[i]), "CLOUD_TPU_TASK_ID": str(i)})
        per_rank.append(r)
    return {"env": env, "per_rank": per_rank}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", action="append", required=True, help="name:chip,chip,...  e.g. policy:0,1,2,3")
    # Not 8470: on Cloud TPU VMs the system agent (`main`) always listens there (P0 crash, 2026-10-24).
    p.add_argument("--port-base", type=int, default=9470, help="first slice-builder port; groups get disjoint ranges")
    p.add_argument("--per-rank-key", default="TPU_VISIBLE_DEVICES",
                   help="env var that selects the rank's chip (overrides the group-level value per rank)")
    p.add_argument("--extra", action="append", default=[], help="KEY=VALUE added to every group's env")
    p.add_argument("--no-libtpu-env", action="store_true", help="omit libtpu's TPU_VISIBLE_CHIPS / TPU_PROCESS_* vars")
    p.add_argument("--json", action="store_true", help="print one JSON object instead of export lines")
    args = p.parse_args()

    extra = dict(kv.split("=", 1) for kv in args.extra)
    out, port = {}, args.port_base
    used = set()
    for g in args.group:
        name, chips_s = g.split(":", 1)
        chips = [int(c) for c in chips_s.split(",") if c != ""]
        if used & set(chips):
            raise SystemExit(f"chips {sorted(used & set(chips))} are assigned to two groups")
        used |= set(chips)
        out[name] = group_spec(chips, port, extra, args.per_rank_key, libtpu=not args.no_libtpu_env)
        port += 16
    if args.json:
        print(json.dumps(out, indent=2))
        return
    for name, spec in out.items():
        print(f"export SKYRL_TPU_GROUP_ENV_{name.upper()}={shlex.quote(json.dumps(spec))}")


if __name__ == "__main__":
    main()

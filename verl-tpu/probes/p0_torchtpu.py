#!/usr/bin/env python3
"""P0: can SkyRL's training layout run on TorchTPU at all? Runs on the TRAINER VM.

Launches the two process groups the CholecT50 run uses (policy and ref, each on
its own chip subset, both at once) with exactly the env the skyrl-tpu patch
sets up (``device.configure_process_env`` + ``SKYRL_TPU_GROUP_ENV_*`` from
``tpu_env.sh``), and checks in every rank, step by step:

  1. device       tensor op on ``"tpu"``
  2. pg_init      ``init_process_group(device.dist_backend())`` = ``cpu:gloo,tpu:tpu_dist``
  3. all_reduce   on tpu and on cpu (gloo), as SkyRL's strategy does
  4. mesh         ``init_device_mesh("tpu", ...)``
  5. fully_shard  FSDP2 on a small model (or ``--model``), MixedPrecisionPolicy bf16
  6. train_step   forward + backward + AdamW step, loss finite
  7. full_state   ``get_model_state_dict(full_state_dict=True, cpu_offload=True)`` (the reload export)
  8. cpu_offload  fully_shard with ``CPUOffloadPolicy`` (what the GPU run used)

    source skyrl_cholect50/tpu_env.sh
    python probes/p0_torchtpu.py                            # tiny model, both groups
    python probes/p0_torchtpu.py --model Qwen/Qwen3-0.6B    # a real HF model
    python probes/p0_torchtpu.py --groups policy            # one group only

Writes ``logs/p0_torchtpu.json``; exit 0 only if every rank passed every stage.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def worker(group: str, rank: int, world: int, master_port: int, model_name: str, out_q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(master_port))
    from skyrl.backends.skyrl_train.utils import device as accel

    accel.configure_process_env(rank=rank, local_rank=rank, world_size=world, group=group)
    res = {"group": group, "rank": rank, "stages": {}, "env": {k: v for k, v in os.environ.items()
                                                             if k.startswith(("TORCH_TPU_", "TPU_"))}}

    def stage(name, fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            res["stages"][name] = {"ok": True, "s": round(time.perf_counter() - t0, 2), "detail": detail}
            return True
        except Exception as e:
            res["stages"][name] = {"ok": False, "s": round(time.perf_counter() - t0, 2),
                                   "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
            return False

    import torch
    import torch.distributed as dist

    ctx = {}

    def s_device():
        x = torch.ones(8, device=accel.current_device()) * (rank + 1)
        return {"sum": x.sum().item(), "device": str(x.device)}

    def s_pg():
        from datetime import timedelta

        dist.init_process_group(backend=accel.dist_backend(), timeout=timedelta(seconds=600))
        return {"backend": accel.dist_backend(), "world": dist.get_world_size()}

    def s_allreduce():
        t = torch.tensor([float(rank + 1)], device=accel.current_device())
        dist.all_reduce(t)
        c = torch.tensor([float(rank + 1)])
        dist.all_reduce(c)
        want = world * (world + 1) / 2
        assert abs(t.item() - want) < 1e-3 and abs(c.item() - want) < 1e-3, (t.item(), c.item(), want)
        return {"tpu": t.item(), "cpu": c.item()}

    def s_mesh():
        from torch.distributed.device_mesh import init_device_mesh

        ctx["mesh"] = init_device_mesh(accel.device_type(), mesh_shape=(world,), mesh_dim_names=["fsdp"])
        return {"mesh": str(ctx["mesh"])}

    def build_model():
        if model_name:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        return torch.nn.Sequential(*[torch.nn.Linear(256, 256) for _ in range(4)])

    def shard(model, offload):
        from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        kw = dict(mesh=ctx["mesh"], mp_policy=mp)
        if offload:
            kw["offload_policy"] = CPUOffloadPolicy(pin_memory=False)
        layers = getattr(getattr(model, "model", None), "layers", None) or list(model.children())
        for layer in layers:
            fully_shard(layer, **kw)
        fully_shard(model, **kw)
        if not offload:
            model.to(accel.current_device())
        return model

    def batch(model):
        if model_name:
            ids = torch.randint(0, 1000, (2, 64), device=accel.current_device())
            return lambda: model(input_ids=ids, labels=ids).loss
        x = torch.randn(4, 256, device=accel.current_device())
        return lambda: model(x).float().pow(2).mean()

    def s_shard():
        ctx["model"] = shard(build_model(), offload=False)
        return {"params": sum(p.numel() for p in ctx["model"].parameters())}

    def s_train():
        model = ctx["model"]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        f = batch(model)
        losses = []
        for _ in range(2):
            loss = f()
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        accel.synchronize()
        assert all(map(lambda v: v == v and abs(v) != float("inf"), losses)), losses
        return {"losses": losses, "mem": accel.mem_get_info()}

    def s_full_state():
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        sd = get_model_state_dict(ctx["model"], options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        n = sum(t.numel() for t in sd.values()) if rank == 0 else 0
        return {"tensors": len(sd), "numel_rank0": n}

    def s_offload():
        model = shard(build_model(), offload=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        loss = batch(model)()
        loss.backward()
        opt.step()
        return {"loss": loss.item()}

    ok = stage("device", s_device) and stage("pg_init", s_pg)
    if ok:
        for name, fn in [("all_reduce", s_allreduce), ("mesh", s_mesh), ("fully_shard", s_shard),
                         ("train_step", s_train), ("full_state", s_full_state), ("cpu_offload", s_offload)]:
            if not stage(name, fn):
                break
    out_q.put(res)
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="policy,ref")
    ap.add_argument("--model", default="", help="HF model id/path; default: tiny MLP")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "p0_torchtpu.json"))
    args = ap.parse_args()
    if os.environ.get("SKYRL_DEVICE") != "tpu":
        sys.exit("source skyrl_cholect50/tpu_env.sh first")

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for group in args.groups.split(","):
        spec = json.loads(os.environ.get(f"SKYRL_TPU_GROUP_ENV_{group.upper()}", "{}"))
        world = len(spec.get("per_rank", [])) or 1
        port = _free_port()
        for r in range(world):
            p = ctx.Process(target=worker, args=(group, r, world, port, args.model, q))
            p.start()
            procs.append(p)
    results = []
    deadline = time.time() + args.timeout
    while len(results) < len(procs) and time.time() < deadline:
        try:
            results.append(q.get(timeout=10))
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
    passed = len(results) == len(procs) and all(all(s["ok"] for s in r["stages"].values()) and len(r["stages"]) == 8
                                                for r in results)
    summary = {"passed": passed, "ranks_expected": len(procs), "ranks_reported": len(results),
               "model": args.model or "tiny-mlp", "results": sorted(results, key=lambda r: (r["group"], r["rank"]))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    for r in summary["results"]:
        line = " ".join(f"{k}={'ok' if v['ok'] else 'FAIL'}" for k, v in r["stages"].items())
        print(f"{r['group']}[{r['rank']}] {line}")
        for k, v in r["stages"].items():
            if not v["ok"]:
                print(f"    {k}: {v['error']}")
    print(f"P0 {'PASSED' if passed else 'FAILED'} -> {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

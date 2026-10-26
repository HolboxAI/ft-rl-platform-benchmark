#!/usr/bin/env python3
"""P0 FSDP2 probe: does fully_shard move CPU params onto `tpu`?

Tests the `_move_states_to_device` patch (TorchTPU tensors are not shallow-copy
compatible with CPU tensors, so fully_shard's `tensor.data = tensor.to(device)`
raises; the patch routes through torch.utils.swap_tensors). Mirrors how verl's
FSDP2 engine builds the mesh (init_device_mesh(device_name, ...)).

Sets the TorchTPU topology env for a single-chip group (what tpu_groups.py
emits for one chip), then runs a tiny fwd/bwd. Run with VERL_PLATFORM=tpu.
"""
import os

# Rendezvous + single-process TorchTPU topology (chip 0, one process).
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ["TORCH_TPU_TOPOLOGY"] = "1,1,1"
os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = "localhost:9470"
os.environ["TPU_VISIBLE_DEVICES"] = "0"
os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
os.environ["TPU_PROCESS_ADDRESSES"] = "localhost:9478"
os.environ["TPU_VISIBLE_CHIPS"] = "0"
os.environ["TPU_PROCESS_PORT"] = "9478"
os.environ["CLOUD_TPU_TASK_ID"] = "0"

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from verl.plugin.platform import get_platform

p = get_platform()
print("platform:", p.device_name, p.vendor_name, p.communication_backend_name())

dist.init_process_group(backend="cpu:gloo,tpu:tpu_dist")
print("process group initialized, rank", dist.get_rank(), "world", dist.get_world_size())


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.fc(x).relu()


m = Tiny()  # params start on CPU
print("before shard, fc.weight device:", m.fc.weight.device)

mesh = init_device_mesh("tpu", mesh_shape=(1,), mesh_dim_names=["fsdp"])
fully_shard(m, mesh=mesh)
print("after shard, fc.weight device:", m.fc.weight.device)

x = torch.randn(8, 64, device="tpu")  # float32, matching the default model dtype
y = m(x).sum()
y.backward()
print("FSDP2 fwd/bwd OK:", y.item(), "| grad is not None:", m.fc.weight.grad is not None)
dist.destroy_process_group()

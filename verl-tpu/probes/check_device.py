#!/usr/bin/env python3
"""P0-style smoke check: TorchTPU device + verl trainer imports.

Confirms (1) TorchTPU registers the ``tpu`` device and a tensor op runs, and
(2) the verl trainer / FSDP engine / HF rollout all import under the TPU
platform. Run with ``VERL_PLATFORM=tpu``.
"""
import importlib

import torch

from verl.plugin.platform import PlatformRegistry, get_platform

print("registered:", PlatformRegistry.registered_names())
p = get_platform()
print("platform:", p.device_name, p.vendor_name, p.communication_backend_name())

for m in [
    "verl.trainer.main_ppo",
    "verl.workers.engine.fsdp.transformer_impl",
    "verl.workers.rollout.hf_rollout",
]:
    importlib.import_module(m)
    print("ok", m)

x = torch.ones(4, device="tpu") * 2
print("tpu tensor op:", x.sum().item(), "| device_count:", p.device_count(), "| current_device:", p.current_device())

"""Go/no-go gate: does the verl platform plugin put a model on `neuron`?

Run this BEFORE any verl training loop. It validates, in order:
  1. verl discovers ``PlatformNeuron`` (``VERL_PLATFORM=neuron``).
  2. ``get_platform().is_available()`` is True.
  3. A model + data move to ``neuron`` and run a forward + backward.
  4. ``torch_neuronx`` reports memory.

This is the Trainium equivalent of the TRL leg's
``trainium2/scripts/gate_test.py``: the cheapest thing that proves the plugin,
the device, and the runtime are wired together. Only after this passes do we
attempt the full verl RL loop (which has its own, larger unknowns — see
VERL_TRAINIUM_ISSUES_LOG.md).
"""
import os

os.environ.setdefault("VERL_PLATFORM", "neuron")
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch
import torch_neuronx  # noqa: F401  registers the `neuron` device

import verl  # noqa: F401  loads the verl.plugins entry points
from verl.plugin.platform import get_platform

plat = get_platform()
print("PLATFORM_DEVICE", plat.device_name, "VENDOR", plat.vendor_name, flush=True)
print("IS_AVAILABLE", plat.is_available(), flush=True)
print("DEVICE_COUNT", plat.device_count(), flush=True)
print("DEVICE_MODULE", plat.device_module, flush=True)

model = torch.nn.Sequential(
    torch.nn.Linear(256, 1024),
    torch.nn.GELU(),
    torch.nn.Linear(1024, 256),
).to("neuron")
x = torch.randn(8, 256, device="neuron")
y = model(x)
loss = y.pow(2).mean()
loss.backward()
print("GATE_FWD_BWD_OK loss=", float(loss), flush=True)
plat.empty_cache()
print("MEM_ALLOCATED_BYTES", torch_neuronx.memory_allocated(), flush=True)
print("GATE_PASS", flush=True)

"""Accelerator backends for the GRPO loop.

Every hardware-specific call the trainer makes goes through a `Backend`, so the
training loop itself is identical on every chip. That is the point of this
benchmark: the same code, the same math, only the device changes.

| backend   | stack                                   | device        | graph model          |
|-----------|-----------------------------------------|---------------|----------------------|
| cpu       | stock PyTorch                           | cpu           | eager (local tests)  |
| cuda      | stock PyTorch                           | cuda          | eager (GPU baseline) |
| neuron    | torch-neuronx: native TorchNeuron       | neuron        | eager / torch.compile|
|           | (or PyTorch/XLA on older images)        | xla           | lazy, mark_step      |
| tpu       | TorchTPU (`torch_tpu`, native backend)  | tpu           | eager / torch.compile|
| tpu-xla   | PyTorch/XLA on Cloud TPU (old path)     | xla           | lazy, mark_step      |

`tpu-xla` exists so the TorchTPU leg can be compared against the previous way of
running PyTorch on TPUs on the same TPU VM.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch

BACKENDS = ("cpu", "cuda", "neuron", "tpu", "tpu-xla")


class Backend:
    name = "cpu"
    lazy = False  # True for XLA-style lazy tensors that need mark_step()

    def __init__(self):
        self.device = torch.device("cpu")

    def mark_step(self):
        """Cut and execute the pending graph (XLA backends only)."""

    def sync(self):
        """Block until all queued device work is done (for honest timings)."""

    def optimizer_step(self, optimizer):
        optimizer.step()
        self.mark_step()

    def memory_gb(self):
        return None

    def describe(self):
        return {"backend": self.name, "device": str(self.device), "torch": torch.__version__}

    @contextmanager
    def timed(self, out: dict, key: str):
        self.sync()
        t0 = time.perf_counter()
        yield
        self.sync()
        out[key] = out.get(key, 0.0) + time.perf_counter() - t0


class CudaBackend(Backend):
    name = "cuda"

    def __init__(self):
        self.device = torch.device("cuda")

    def sync(self):
        torch.cuda.synchronize()

    def memory_gb(self):
        return torch.cuda.max_memory_allocated() / 1e9


class XlaBackend(Backend):
    """PyTorch/XLA lazy-tensor backend. Shared by Trainium (torch-neuronx) and
    TPU (torch_xla). Graphs are traced lazily and compiled on mark_step(); every
    distinct tensor shape is a new compile, which is why the rollout code keeps
    all shapes static."""

    name = "xla"
    lazy = True

    def __init__(self):
        import torch_xla.core.xla_model as xm

        self.xm = xm
        self.device = xm.xla_device()

    def mark_step(self):
        self.xm.mark_step()

    def sync(self):
        self.xm.mark_step()
        self.xm.wait_device_ops()

    def optimizer_step(self, optimizer):
        # xm.optimizer_step also all-reduces gradients when running data-parallel.
        self.xm.optimizer_step(optimizer)
        self.xm.mark_step()

    def describe(self):
        d = super().describe()
        try:
            import torch_xla

            d["torch_xla"] = torch_xla.__version__
        except Exception:
            pass
        return d


class NeuronNativeBackend(Backend):
    """TorchNeuron: torch-neuronx as a native PyTorch backend (Neuron >= 2.27).

    Registers device "neuron" through PyTorch's PrivateUse1 mechanism; eager by
    default, torch.compile supported, no torch_xla and no mark_step. The
    torch-neuronx 2.14 DLAMI (checked 2026-10-22) ships this and no torch_xla.
    Synchronisation and memory stats go through torch.accelerator, falling back
    to a torch.neuron module if the installed version exposes one.
    """

    name = "neuron"
    lazy = False

    def __init__(self):
        import torch_neuronx  # noqa: F401  (registers the "neuron" device)

        self.device = torch.device("neuron")
        torch.ones(1, device=self.device).sum().item()  # fail here, not mid-run, if the device is unusable

    def _mod(self):
        return getattr(torch, "neuron", None)

    def sync(self):
        acc = getattr(torch, "accelerator", None)
        if acc is not None and hasattr(acc, "synchronize"):
            acc.synchronize()
        elif self._mod() is not None and hasattr(self._mod(), "synchronize"):
            self._mod().synchronize()

    def memory_gb(self):
        for mod in (self._mod(), getattr(torch, "accelerator", None)):
            fn = getattr(mod, "max_memory_allocated", None) if mod is not None else None
            if fn is not None:
                try:
                    return fn() / 1e9
                except Exception:
                    pass
        return None

    def describe(self):
        d = super().describe()
        d["neuron_mode"] = "native"
        try:
            import torch_neuronx

            d["torch_neuronx"] = getattr(torch_neuronx, "__version__", "unknown")
        except Exception:
            pass
        return d


class NeuronXlaBackend(XlaBackend):
    """torch-neuronx on PyTorch/XLA (Neuron < 2.32; images that still ship torch_xla)."""

    name = "neuron"

    def __init__(self):
        import torch_neuronx  # noqa: F401  (registers the Neuron PJRT plugin)

        super().__init__()

    def describe(self):
        d = super().describe()
        d["neuron_mode"] = "xla"
        try:
            import torch_neuronx

            d["torch_neuronx"] = getattr(torch_neuronx, "__version__", "unknown")
        except Exception:
            pass
        return d


def NeuronBackend():
    """torch-neuronx mode from NEURON_MODE: native | xla | auto (default).
    auto = native TorchNeuron, falling back to PyTorch/XLA only if native fails
    and torch_xla is installed (older Neuron images)."""
    import importlib.util
    import os

    mode = os.environ.get("NEURON_MODE", "auto")
    if mode == "native":
        return NeuronNativeBackend()
    if mode == "xla":
        return NeuronXlaBackend()
    try:
        return NeuronNativeBackend()
    except Exception:
        if importlib.util.find_spec("torch_xla") is None:
            raise
        return NeuronXlaBackend()


class TpuXlaBackend(XlaBackend):
    name = "tpu-xla"


class TorchTpuBackend(Backend):
    """Native TorchTPU backend (`pip install torch_tpu`, device "tpu").

    TorchTPU runs eagerly by default and supports torch.compile; there is no
    mark_step. The synchronize call is looked up defensively because the API is
    new — check it against the TorchTPU docs for the installed version.
    """

    name = "tpu"

    def __init__(self):
        import torch_tpu  # noqa: F401  (registers the "tpu" device)

        self.device = torch.device("tpu")

    def sync(self):
        mod = getattr(torch, "tpu", None)
        fn = getattr(mod, "synchronize", None) if mod is not None else None
        if fn is not None:
            fn()

    def memory_gb(self):
        mod = getattr(torch, "tpu", None)
        fn = getattr(mod, "max_memory_allocated", None) if mod is not None else None
        return fn() / 1e9 if fn is not None else None

    def describe(self):
        d = super().describe()
        try:
            import torch_tpu

            d["torch_tpu"] = getattr(torch_tpu, "__version__", "unknown")
        except Exception:
            pass
        return d


def get_backend(name: str) -> Backend:
    if name == "cpu":
        return Backend()
    if name == "cuda":
        return CudaBackend()
    if name == "neuron":
        return NeuronBackend()
    if name == "tpu":
        return TorchTpuBackend()
    if name == "tpu-xla":
        return TpuXlaBackend()
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")

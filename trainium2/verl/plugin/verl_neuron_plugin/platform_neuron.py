"""TorchNeuron platform backend for verl.

Registers verl's ``PlatformBase`` for the AWS Trainium ``neuron`` device so
verl runs on Trainium2/Trn3 through TorchNeuron (the native PyTorch backend,
not the PyTorch/XLA ``torch-neuronx`` product).

verl auto-loads this via the setuptools entry-point group ``verl.plugins``
(``pyproject.toml``), so no verl source tree change is required. Set
``VERL_PLATFORM=neuron`` to force selection.
"""

from __future__ import annotations

import contextlib
import os
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform import PlatformBase, PlatformRegistry


def _resolve_device_module() -> ModuleType:
    """Return the module exposing the standard device API (is_available, etc.).

    ``torch-neuronx`` registers the ``neuron`` device via PyTorch PrivateUse1.
    After ``torch.utils.rename_privateuse1_backend("neuron")`` the standard
    device namespace lives at ``torch.neuron``; the runtime helpers
    (``empty_cache``/``synchronize``/``memory_allocated``) live on
    ``torch_neuronx``. Prefer ``torch.neuron`` when present.
    """
    try:
        import torch_neuronx  # noqa: F401  registers the `neuron` device
    except ImportError:
        pass
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "is_available"):
        return torch.neuron
    import torch_neuronx
    return torch_neuronx


def _runtime() -> ModuleType:
    """Return the module holding runtime helpers (empty_cache, synchronize)."""
    try:
        import torch_neuronx
        return torch_neuronx
    except ImportError:
        return _resolve_device_module()


@PlatformRegistry.register(platform="neuron")
class PlatformNeuron(PlatformBase):
    """Trainium2/Trn3 platform via TorchNeuron (native PyTorch)."""

    @property
    def device_name(self) -> str:
        return "neuron"

    @property
    def vendor_name(self) -> str:
        return "aws"

    @property
    def device_module(self) -> ModuleType:
        return _resolve_device_module()

    def is_available(self) -> bool:
        try:
            return bool(self.device_module.is_available())
        except Exception:
            return False

    def current_device(self) -> int:
        try:
            return int(self.device_module.current_device())
        except Exception:
            return 0

    def device_count(self) -> int:
        try:
            return int(self.device_module.device_count())
        except Exception:
            cores = os.getenv("NEURON_RT_VISIBLE_CORES", "0")
            n = len([c for c in cores.split(",") if c.strip()])
            return n or 1

    def set_device(self, device_index: int) -> None:
        # NeuronCore selection is via NEURON_RT_VISIBLE_CORES, not set_device.
        # No-op keeps verl's device-select calls harmless.
        return

    def synchronize(self, device_index: Optional[int] = None) -> None:
        try:
            _runtime().synchronize()
        except Exception:
            pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def set_allocator_settings(self, settings: str) -> None:
        return

    def empty_cache(self) -> None:
        try:
            _runtime().empty_cache()
        except Exception:
            pass

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    def communication_backend_name(self) -> str:
        # TorchNeuron's torch.distributed backend (verify on-instance; the
        # PrivateUse1 convention registers the backend under the device name).
        return "neuron"

    def visible_devices_envvar(self) -> str:
        return "NEURON_RT_VISIBLE_CORES"

    @contextlib.contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        return

    def profiler_stop(self) -> None:
        return

    def ray_resource_name(self) -> str:
        return "neuron"

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES"]

    def is_ipc_supported(self) -> bool:
        return False

    def cudart(self) -> Any:
        return None

    def apply_model_patches(self, model_type: str) -> None:
        """Device-specific model patches, called by the engine before model init.

        Qwen3.5's vision tower uses ``torch.repeat_interleave`` with a dynamic
        ``repeats`` tensor; neuronx-cc cannot lower that op. Compute the small
        ``(total_thw, n_taps)`` index/weight tensors on CPU and move them back.
        This is the same patch as the TRL leg
        (``trainium2/scripts/grpo_train.py:_patch_vision_interp``).
        """
        try:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        except Exception:
            return

        _orig = mq.get_vision_interpolation_indices_and_weights

        def _patched(grid_thw, *args, **kwargs):
            if grid_thw.device.type == "neuron":
                indices, weights = _orig(grid_thw.to("cpu"), *args, **kwargs)
                return indices.to(grid_thw.device), weights.to(grid_thw.device)
            return _orig(grid_thw, *args, **kwargs)

        mq.get_vision_interpolation_indices_and_weights = _patched

"""TorchTPU platform backend for verl.

Registers verl's ``PlatformBase`` for the Google Cloud TPU ``tpu`` device so
verl runs on Cloud TPU v6e (Trillium) through TorchTPU (``torch_tpu``, the
native PyTorch backend registered under PrivateUse1).

verl auto-loads this via the setuptools entry-point group ``verl.plugins``
(``pyproject.toml``), so no verl source tree change is required. Set
``VERL_PLATFORM=tpu`` to force selection.

TorchTPU notes (torch_tpu 0.2.x):

* One process per chip; a chip can be opened by one process only.
* ``torch.accelerator.*`` is the supported API (``torch.tpu`` is deprecated).
* With ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` (set on the VM so DataLoader workers
  don't grab chips), ``import torch_tpu`` leaves the backend unnamed; only
  ``torch_tpu._loader.load()`` registers ``"tpu"``.
"""

from __future__ import annotations

import contextlib
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform import PlatformBase, PlatformRegistry

_FSDP2_PATCHED = False
_MROPE_PATCHED = False


def _load_torch_tpu() -> None:
    """Register the ``"tpu"`` device (idempotent).

    With autoload off, ``import torch_tpu`` alone leaves the backend unnamed
    (``"privateuseone"``); the loader registers ``"tpu"``.
    """
    if torch._C._get_privateuse1_backend_name() == "tpu":
        return
    import torch_tpu  # noqa: F401
    from torch_tpu._loader import load

    load()


def _patch_accelerator_api() -> None:
    """Add torch.cuda-style helpers verl expects (mem_get_info).

    TorchTPU's ``torch.accelerator`` exposes ``get_memory_info``, not the
    ``mem_get_info`` name verl calls via ``get_torch_device().mem_get_info()``.
    """
    acc = torch.accelerator
    if hasattr(acc, "mem_get_info"):
        return

    def mem_get_info(device=None):
        fn = getattr(acc, "get_memory_info", None)
        if fn is None:
            return (0, 0)
        try:
            info = fn()
            if isinstance(info, (tuple, list)) and len(info) == 2:
                return int(info[0]), int(info[1])
            if isinstance(info, dict):
                total = int(info.get("bytes_limit") or info.get("total") or 0)
                used = int(info.get("bytes_in_use") or info.get("used") or 0)
                return max(total - used, 0), total
        except Exception:
            pass
        return (0, 0)

    acc.mem_get_info = mem_get_info


def _resolve_device_module() -> ModuleType:
    """Return the module exposing the standard device API (is_available, etc.)."""
    _load_torch_tpu()
    _patch_accelerator_api()
    return torch.accelerator


def _patch_fsdp2_move_states() -> None:
    """Let FSDP2 move CPU params onto ``tpu`` via ``torch.utils.swap_tensors``.

    ``fully_shard`` moves each CPU parameter with ``tensor.data = tensor.to(device)``
    (``torch/distributed/fsdp/_fully_shard/_fsdp_init.py:_move_states_to_device``).
    TorchTPU tensors are not shallow-copy compatible with CPU tensors, so that
    assignment raises an "incompatible tensor type" error. This routes every
    incompatible tensor through ``torch.utils.swap_tensors``; compatible tensors
    (all of them on CUDA) keep the original code path.
    """
    global _FSDP2_PATCHED
    if _FSDP2_PATCHED:
        return
    import itertools

    import torch.nn as nn
    from torch.distributed.fsdp._fully_shard import _fsdp_init

    original = _fsdp_init._move_states_to_device

    def _move_states_to_device(params, buffers, device):
        rest_params, rest_buffers = [], []
        for tensor, bucket in itertools.chain(
            ((p, rest_params) for p in params), ((b, rest_buffers) for b in buffers)
        ):
            if tensor.device == device or tensor.device.type == "meta" or type(tensor).__name__ == "DTensor":
                bucket.append(tensor)
                continue
            with torch.no_grad():
                moved = tensor.to(device)
            if torch._has_compatible_shallow_copy_type(tensor, moved):
                bucket.append(tensor)
                continue
            if isinstance(tensor, nn.Parameter):
                moved = nn.Parameter(moved, requires_grad=tensor.requires_grad)
            torch.utils.swap_tensors(tensor, moved)
        original(rest_params, rest_buffers, device)

    _fsdp_init._move_states_to_device = _move_states_to_device
    _FSDP2_PATCHED = True


def _apply_interleaved_mrope(self, freqs, mrope_section):
    """Same values as transformers' version, without in-place writes.

    TorchTPU 0.2 aborts on in-place operations between views of one tensor
    (``inversion.cc:112``), including the strided slice assignment in
    ``apply_interleaved_mrope``. ``torch.where`` builds the interleaved layout
    with no in-place writes.
    """
    d = freqs.shape[-1]
    col = torch.arange(d, device=freqs.device)
    use_h = (col % 3 == 1) & (col < mrope_section[1] * 3)
    use_w = (col % 3 == 2) & (col < mrope_section[2] * 3)
    return torch.where(use_h, freqs[1], torch.where(use_w, freqs[2], freqs[0]))


def _patch_qwen35_models() -> None:
    """Patch Qwen3.5's mRoPE + vision RoPE for TorchTPU (idempotent)."""
    global _MROPE_PATCHED
    if _MROPE_PATCHED:
        return
    import inspect

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    except ImportError:
        return
    for _, cls in inspect.getmembers(m, inspect.isclass):
        if "apply_interleaved_mrope" in cls.__dict__:
            cls.apply_interleaved_mrope = _apply_interleaved_mrope
    vision = getattr(m, "Qwen3_5VisionRotaryEmbedding", None)
    if vision is not None:

        def _vision_forward(self, position_ids):
            # inv_freq is a non-persistent buffer that FSDP can leave on CPU;
            # move it to the position_ids device before the multiply.
            inv_freq = self.inv_freq.to(position_ids.device)
            return (position_ids.unsqueeze(-1) * inv_freq).flatten(1)

        vision.forward = _vision_forward
    _MROPE_PATCHED = True


@PlatformRegistry.register(platform="tpu")
class PlatformTPU(PlatformBase):
    """Google Cloud TPU (Trillium) platform via TorchTPU."""

    def __init__(self) -> None:
        # Torch-level patches must be in place before FSDP2 init / model build.
        _patch_fsdp2_move_states()

    @property
    def device_name(self) -> str:
        return "tpu"

    @property
    def vendor_name(self) -> str:
        return "google"

    @property
    def device_module(self) -> ModuleType:
        return _resolve_device_module()

    def is_available(self) -> bool:
        try:
            return bool(self.device_module.is_available())
        except Exception:
            return False

    def current_device(self) -> int:
        # One process per chip: each process owns exactly its own device.
        return 0

    def device_count(self) -> int:
        # One process per chip: a TPU process sees exactly one device.
        return 1

    def set_device(self, device_index: int) -> None:
        # Device selection is via the topology env (TPU_VISIBLE_DEVICES) set
        # before the backend initialises, not per call.
        return

    def synchronize(self, device_index: Optional[int] = None) -> None:
        try:
            self.device_module.synchronize()
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
            self.device_module.empty_cache()
        except Exception:
            pass

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    def communication_backend_name(self) -> str:
        # TorchTPU's c10d backend; the full init string is "cpu:gloo,tpu:tpu_dist".
        return "tpu_dist"

    def visible_devices_envvar(self) -> str:
        return "TPU_VISIBLE_DEVICES"

    @contextlib.contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        return

    def profiler_stop(self) -> None:
        return

    def ray_resource_name(self) -> str:
        # Map TPU chips to Ray "GPU" slots (ray start --num-gpus=N), as the
        # SkyRL TPU run did.
        return "GPU"

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_DEVICES"]

    def is_ipc_supported(self) -> bool:
        return False

    def cudart(self) -> Any:
        return None

    def apply_model_patches(self, model_type: str) -> None:
        _patch_qwen35_models()

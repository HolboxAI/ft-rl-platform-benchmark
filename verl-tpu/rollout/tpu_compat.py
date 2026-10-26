"""TorchTPU workarounds for transformers' Qwen3.5 code, shared by the rollout HF engine.

``apply_interleaved_mrope`` writes into ``freqs[0]`` (a view of ``freqs``) from other
slices of the same tensor. TorchTPU 0.2 aborts on in-place operations between
overlapping views (``inversion.cc:112``; seen with batched, left-padded ``generate`` on
v6e, 2026-10-24), including the strided slice assignment itself, so the replacement builds
the interleaved layout with ``torch.where`` instead: identical values, no in-place writes.
The skyrl-tpu patch applies the same fix in the trainer
(``skyrl/backends/skyrl_train/utils/device.py:patch_qwen35_mrope_for_tpu``).
"""

_PATCHED = False


def load_torch_tpu() -> None:
    """Register the ``"tpu"`` device even when ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` is inherited."""
    import torch

    if torch._C._get_privateuse1_backend_name() != "tpu":  # `import torch_tpu` alone leaves it unnamed
        import torch_tpu  # noqa: F401
        from torch_tpu._loader import load

        load()


def _apply_interleaved_mrope(self, freqs, mrope_section):
    # Same result as transformers' version, without in-place writes: TorchTPU 0.2 aborts on its
    # strided in-place assignment between views of one tensor (inversion.cc:112).
    import torch

    d = freqs.shape[-1]
    col = torch.arange(d, device=freqs.device)
    use_h = (col % 3 == 1) & (col < mrope_section[1] * 3)
    use_w = (col % 3 == 2) & (col < mrope_section[2] * 3)
    return torch.where(use_h, freqs[1], torch.where(use_w, freqs[2], freqs[0]))


def patch_qwen35_mrope() -> int:
    """Replace ``apply_interleaved_mrope`` on every Qwen3.5 class that defines it. Returns the count."""
    global _PATCHED
    if _PATCHED:
        return 0
    import inspect

    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    n = 0
    for _, cls in inspect.getmembers(m, inspect.isclass):
        if "apply_interleaved_mrope" in cls.__dict__:
            cls.apply_interleaved_mrope = _apply_interleaved_mrope
            n += 1
    _PATCHED = True
    return n

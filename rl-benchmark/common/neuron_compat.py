"""TorchNeuron compatibility shim for Qwen3.5 (transformers 5.19) — index bookkeeping on CPU.

Problem (run trl-trn-20261022-1141, 2026-10-22): the first `generate` on the
CholecT50 checkpoint failed inside the TorchNeuron compiler:

    transformers/vision_utils.py: get_vision_interpolation_indices_and_weights
        heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    TorchMlirCompilerError: 'stablehlo.concatenate' op inferred type(s) 'tensor<2xi64>'
    are incompatible with return type(s) of operation 'tensor<1xi64>'

`repeat_interleave` with a tensor of repeats has a data-dependent output shape,
which TorchNeuron's Torch-MLIR -> StableHLO lowering does not handle.
transformers documents these `grid_thw` helpers as using "untraceable ops
(repeat_interleave, .tolist(), nonzero(), loops)" and meant to be precomputed.

Fix: run those helpers (and Qwen3.5's mRoPE index computation, which is the
same kind of code) on CPU when their inputs live on the neuron device, then move
the small integer/float results back. Model math stays on Trainium; only
O(num_patches) index tensors cross to the host, once per forward.

Call `install()` before building the model. No-op off Neuron. Used by hf-trl/ (TRL) and
common/grpo (our GRPO loop, --backend neuron).
"""
from __future__ import annotations

import functools
import inspect

import torch

PATCHED: list = []


def _to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def _target_device(args, kwargs):
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, torch.Tensor):
            return v.device
    dev = kwargs.get("device")
    return torch.device(dev) if dev is not None else None


def on_cpu(fn):
    """Run `fn` on CPU copies of its tensor args when they live on neuron; move the result back."""
    if getattr(fn, "_neuron_cpu", False):
        return fn
    has_device_kw = "device" in inspect.signature(fn).parameters

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        target = _target_device(args, kwargs)
        if target is None or target.type != "neuron":
            return fn(*args, **kwargs)
        # Leave `self` (first positional of a method) and the shared `kwargs` dict object alone:
        # helpers pop precomputed keys from the caller's kwargs dict, so it must not be copied.
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        cpu_kwargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        if has_device_kw and "device" in kwargs:
            cpu_kwargs["device"] = "cpu"
        return _to(fn(*cpu_args, **cpu_kwargs), target)

    wrapped._neuron_cpu = True
    return wrapped


def install(verbose: bool = True) -> list:
    """Patch transformers' vision index helpers and Qwen3.5's rope-index methods. Idempotent."""
    import transformers.vision_utils as vu

    names = [n for n in dir(vu) if n.startswith("get_vision_") and callable(getattr(vu, n))]
    for n in names:
        setattr(vu, n, on_cpu(getattr(vu, n)))
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
    except ImportError:
        q = None
    if q is not None:
        # modeling_qwen3_5 imported the helpers by name: patch its references too.
        for n in names:
            if hasattr(q, n):
                setattr(q, n, getattr(vu, n))
        # get_placeholder_mask: on neuron it counted 203 image tokens for a frame with 405
        # (run trl-trn-20261022-1141-r2) although the same count is right standalone; computing the
        # mask on CPU sidesteps it (inputs_embeds is copied once per forward, ~10 MB).
        for cls_name, meth in (("Qwen3_5Model", "get_rope_index"), ("Qwen3_5Model", "get_vision_position_ids"),
                               ("Qwen3_5Model", "get_placeholder_mask")):
            cls = getattr(q, cls_name, None)
            if cls is not None and hasattr(cls, meth):
                setattr(cls, meth, on_cpu(getattr(cls, meth)))
                names.append(f"{cls_name}.{meth}")
    PATCHED[:] = names
    if verbose:
        print(f"[neuron_compat] CPU index helpers: {', '.join(names)}", flush=True)
    return names

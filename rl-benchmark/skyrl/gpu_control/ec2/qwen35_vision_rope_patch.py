"""Lazy patch for Qwen3.5's vision RoPE under FSDP cpu_offload, for every process
in SkyRL's venv (API server, engine, Ray workers).

Same fix as reference_gpu/entrypoint.py:apply_vision_rope_device_patch. That run
applied it inside its own Ray entrypoint. The Tinker-API server has no such hook,
so job.sh installs this module next to a `.pth` file in the venv's site-packages.
The import hook patches `Qwen3_5VisionRotaryEmbedding.forward` the moment
transformers' modeling_qwen3_5 module is imported, so no other module is
imported eagerly at interpreter start.
"""
import importlib.abc
import importlib.util
import sys

_TARGET = "transformers.models.qwen3_5.modeling_qwen3_5"


def _patch(mod):
    cls = getattr(mod, "Qwen3_5VisionRotaryEmbedding", None)
    if cls is None or getattr(cls, "_cholec_patched", False):
        return

    def forward(self, position_ids):
        inv_freq = self.inv_freq.to(position_ids.device)
        return (position_ids.unsqueeze(-1) * inv_freq).flatten(1)

    cls.forward = forward
    cls._cholec_patched = True
    print("[cholec] patched Qwen3_5VisionRotaryEmbedding.forward (device-safe inv_freq)", file=sys.stderr, flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)  # find the real spec without recursing into ourselves
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())

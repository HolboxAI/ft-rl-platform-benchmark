"""verl platform plugin for Google Cloud TPU (TorchTPU).

Importing this package registers ``PlatformTPU`` under the platform name
``tpu`` and re-registers verl's FSDP engines for the tpu device. verl
discovers it via the setuptools entry-point group ``verl.plugins`` (see
``../pyproject.toml``); no verl source edit is needed.
"""

from .platform_tpu import PlatformTPU  # noqa: F401  (fires @PlatformRegistry.register)


def _register_engines_for_tpu():
    """Register verl's FSDP engines for the tpu device.

    verl's EngineRegistry maps (device, vendor) -> engine class, and the FSDP
    engines are registered under ``device=["cuda", "npu"]``. Add ``tpu`` so the
    lookup for ``device='tpu', backend='fsdp2'`` resolves to the same FSDP
    engine (which then uses ``get_device_name()`` == "tpu" internally).
    """
    from verl.workers.engine import EngineRegistry
    from verl.workers.engine.fsdp.transformer_impl import (
        FSDPEngineWithLMHead,
        FSDPEngineWithValueHead,
    )

    EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="tpu")(FSDPEngineWithLMHead)
    EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="tpu")(FSDPEngineWithValueHead)


_register_engines_for_tpu()

__all__ = ["PlatformTPU"]

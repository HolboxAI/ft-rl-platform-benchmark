"""verl platform plugin for Google Cloud TPU (TorchTPU).

Importing this package registers ``PlatformTPU`` under the platform name
``tpu``. verl discovers it via the setuptools entry-point group
``verl.plugins`` (see ``../pyproject.toml``); no verl source edit is needed.
"""

from .platform_tpu import PlatformTPU  # noqa: F401  (fires @PlatformRegistry.register)

__all__ = ["PlatformTPU"]

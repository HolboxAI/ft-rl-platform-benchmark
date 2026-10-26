"""verl platform plugin for AWS Trainium (TorchNeuron).

Importing this package registers ``PlatformNeuron`` under the platform name
``neuron``. verl discovers it via the setuptools entry-point group
``verl.plugins`` (see ``../pyproject.toml``); no verl source edit is needed.
"""

from .platform_neuron import PlatformNeuron  # noqa: F401  (fires @PlatformRegistry.register)

__all__ = ["PlatformNeuron"]

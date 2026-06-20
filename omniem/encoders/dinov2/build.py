"""Build a backbone module from an ``arch`` name.

This is a *thin* dispatcher: look up the arch in :data:`ARCH_REGISTRY` and call
its zero-arg factory. There is no user knob for any build arg — every
hyperparameter is owner-frozen inside the factory function.
"""

from __future__ import annotations

import torch.nn as nn

from omniem.encoders.registry import arch_info


def build(arch: str) -> nn.Module:
    """Construct the bare backbone module from an ``arch`` name.

    Args:
        arch: The :data:`ARCH_REGISTRY` key selecting the registered factory
            (e.g. ``"emdinov1"``).

    Returns:
        The instantiated ``nn.Module`` — random init, eval mode NOT set. Loading
        pretrained weights is the caller's responsibility.

    Raises:
        ConfigError: If ``arch`` is not registered (raised by :func:`arch_info`,
            which lists the available archs).
    """
    return arch_info(arch).factory()


__all__ = ["build"]

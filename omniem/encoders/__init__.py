"""omniem.encoders — the public encoder surface.

Exports :class:`EMEncoder`. The arch-keyed :data:`ARCH_REGISTRY` is an
implementation detail; users only see arch names through their config files / registry
entries.
"""

from omniem.encoders.base import EMEncoder

__all__ = ["EMEncoder"]

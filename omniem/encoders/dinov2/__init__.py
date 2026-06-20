"""DINOv2 family. The package subpath that owns the vendored backbone +
the omniem-driven forward loop. The class itself is in :mod:`.backbone`; the
construction helper is :mod:`.build`; the no-OOP forward function is :mod:`.forward`.
"""

from omniem.encoders.dinov2.build import build
from omniem.encoders.dinov2.forward import forward_features

__all__ = ["build", "forward_features"]

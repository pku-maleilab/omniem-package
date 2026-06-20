"""omniem model layer.

:class:`OmniEMV1Net` is the bare ``nn.Module`` (UNETR head + STAdapter z-fusion
over a bare DinoVisionTransformer); :class:`OmniEM` (``base.py``) is the
user-facing wrapper that handles config + weights I/O.

Shape rule: XY must be square AND a multiple of :data:`XY_SIDE_DIVISOR`
(= ``lcm(14, 16) == 112``, the lcm of the ViT patch size and the omniem_patch
constant). :class:`OmniEM.predict` raises
:class:`omniem.errors.InputContractError` on a bad shape.
"""

from __future__ import annotations

from omniem.models.adapter import STAdapter
from omniem.models.omniemv1_net import OmniEMV1Net
from omniem.models.registry import (
    MODEL_ARCH_REGISTRY,
    ModelArchInfo,
    list_models,
    model_arch_info,
)
from omniem.models.upsample import UpBlock

# XY side divisor. The legacy resize-to-emdino path asserts ``X * vit_patch / 16`` is
# integer; with vit_patch=14 and omniem_patch=16 the smallest XY side that satisfies
# BOTH grid constraints is lcm(14, 16) = 112. Production heads (e.g. 448 = 4*112)
# inherit the rule.
XY_SIDE_DIVISOR: int = 112

__all__ = [
    "MODEL_ARCH_REGISTRY",
    "ModelArchInfo",
    "OmniEMV1Net",
    "STAdapter",
    "UpBlock",
    "XY_SIDE_DIVISOR",
    "list_models",
    "model_arch_info",
]

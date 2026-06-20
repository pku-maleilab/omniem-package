"""Shared primitives for the input/output stage split.

This module provides two cross-layer building blocks reused by both the encoder
(:mod:`omniem.encoders.base`) and the model (:mod:`omniem.models.base`):

* :class:`Prepared` — the frozen carrier returned by ``apply_input``. It bundles
  the canonical tensor with the metadata ``predict`` needs to invert the
  conform/un-fold round-trip (``axes``, ``conform``, ``orig_yx``,
  ``pad_or_scale``, ``B``, ``Z``, ``stride``). One dataclass is shared across
  layers because the **stride** (14 for the encoder, 112 for the model) acts
  as the discriminator — the conform/un-conform code reads ``prepared.stride``.
  Layer-misuse (feeding an encoder-prepared 4D tensor into the model, or vice
  versa) is caught by the ndim + ``C == in_chans`` + ``Z == img_z`` validation
  the consumer runs before compute.
* :func:`channel_axis_from_axes` — the single source of truth for "where does
  the channel axis sit in a tensor that was produced by ``predict``?" Used by
  both :meth:`OmniEM.apply_output` and :class:`OmniEM._to_caller_axes`. Rule:
  ``c`` sits right after ``b`` if present, else at axis 0 — the same rule the
  shipped ``_to_caller_axes`` already encodes for output reshaping.

Keeping these two utilities here (rather than under ``omniem/models/``) lets
the encoder import them without depending on the model layer, and avoids the
``from omniem.models import output as _output`` style late-import dance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

# Conform vocabulary. The model accepts all three; the encoder accepts only
# ``'resize'`` / ``'strict'`` (``'pad'`` would leave padded-region tokens in
# patch/inner with no restore).
ConformMode = Literal["pad", "resize", "strict"]


@dataclass(frozen=True)
class Prepared:
    """The output of ``apply_input`` — a canonical tensor + un-conform metadata.

    Attributes:
        tensor: The canonical tensor consumed by ``compute``. Encoder layout =
            ``[B*Z, in_chans, S, S]`` (4D, Z folded into the batch); model
            layout = ``[B, C, Y, X', Z]`` (5D, Z kept for the CNN stem / z-fusion;
            ``C == in_chans`` after channel synthesis). The shape's ndim is the
            primary cross-layer discriminator.
        axes: The caller's original axes string (e.g. ``'yx'``, ``'zyx'``,
            ``'bzyx'``). ``predict`` un-folds back to this layout via
            ``_to_caller_axes``.
        conform: The conform mode that was applied. Empty / ``'strict'`` means
            "no transform was applied"; the un-conform is a no-op.
        orig_yx: The caller's pre-conform XY ``(Y, X)``. Used by ``predict`` to
            crop (``'pad'``) or resize-back (``'resize'``) the logits to the
            caller's original spatial size.
        pad_or_scale: Conform-mode-specific recovery info — for ``'pad'`` the
            bottom/right pad extents ``(pad_y, pad_x)`` (always
            ``(S*-Y, S*-X)`` under bottom/right padding, so cropping is just
            ``[..., :Y, :X]``); for ``'resize'`` it carries the conformed
            square side ``S*`` (or interpolation parameters) so the resize-back
            can mirror the forward resize. Free-form to keep the dataclass
            stable as we add modes — consumers know what to read for their
            ``conform`` value.
        B: The caller-side batch count. ``predict`` consults this together with
            ``axes`` to decide whether the output layout has a leading ``b``.
        Z: The Z extent the caller passed in (``1`` when ``axes`` had no ``z``).
            Encoder-prepared tensors carry the pre-fold ``Z`` so the
            return-flag assembly can un-fold patch / inner tokens when the
            standalone encoder is used directly.
        stride: The arch divisor that was enforced on conform (14 emdinov1, 112
            omniemv1). The conform / un-conform math reads this — passing a
            ``Prepared`` between layers with the wrong stride is caught by the
            consumer's validation, but the value also lets the un-conform code
            stay layer-agnostic.
    """

    tensor: torch.Tensor
    axes: str
    conform: ConformMode
    orig_yx: tuple[int, int]
    pad_or_scale: dict[str, Any] = field(default_factory=dict)
    B: int = 1
    Z: int = 1
    stride: int = 1


def channel_axis_from_axes(axes: str) -> int:
    """Return the index of the ``c`` axis in a tensor produced by ``predict``.

    Single source of truth shared by :meth:`OmniEM.apply_output` and the
    model's ``_to_caller_axes`` — both must agree on where the channel sits in
    a caller-axes output tensor so a wrong rule doesn't silently mis-locate
    the channel.

    ``predict`` always inserts a ``c`` axis right
    after a leading ``b`` if present (e.g. ``'byx' → 'bcyx'``,
    ``'bzyx' → 'bczyx'``), else at axis 0 (``'yx' → 'cyx'``,
    ``'zyx' → 'czyx'``). The caller's input ``axes`` may or may not contain a
    ``c`` itself — we ignore that and consult only whether ``b`` is the
    leading axis, because ``predict``'s output is the rule we're inverting.

    Args:
        axes: The axes string the caller passed to ``predict``. Lowercase.

    Returns:
        ``1`` if ``axes`` starts with ``'b'``, else ``0``.
    """
    if not isinstance(axes, str):
        raise TypeError(f"axes must be a str (got {type(axes).__name__})")
    return 1 if axes.startswith("b") else 0


__all__ = ["ConformMode", "Prepared", "channel_axis_from_axes"]

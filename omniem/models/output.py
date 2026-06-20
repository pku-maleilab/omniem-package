"""Internal output-stage transforms.

The output stage is owned by the model (:meth:`omniem.models.base.OmniEM.apply_output`),
gated by ``config.task_type``. The math lives here as **internal** helpers so the
model method and the CLI share one implementation (drift guard). There is no
public free function in this module — a standalone
``apply_output(out_channels=, output=)`` would be a footgun.

Two transforms:

* :func:`_apply_image2image` — sigmoid → clamp[0,1] → ·(2ⁿ-1) → round → ``dtype``,
  then **squeeze** the C=1 channel (the single-channel collapse).
* :func:`_apply_image2label` — argmax over the channel axis → class map as
  ``dtype``. There is no 1-channel threshold branch: image2label always has
  ``out_channels >= 2`` by construction.

Both **collapse the channel axis** — the result has no ``c`` dim.

uint scaling is ``clamp[0,1]·(2ⁿ-1)→round`` (full scale 255 / 65535).
"""

from __future__ import annotations

from typing import Literal

import torch

from omniem.errors import InputContractError

# Accepted dtype names for the integer output (the only knob exposed).
LabelDtype = Literal["uint8", "uint16"]

# Integer-target full-scale ranges. ``uint8`` is the standard 8-bit display
# range; ``uint16`` mirrors tifffile's default 16-bit range. ``torch.uint16``
# requires torch>=2.3 (pinned in ``pyproject.toml``).
_UINT_INFO: dict[str, tuple[int, torch.dtype]] = {
    "uint8": (255, torch.uint8),
    "uint16": (65535, torch.uint16),
}


def _resolve_dtype(name: str) -> tuple[int, torch.dtype]:
    """Map ``'uint8'`` / ``'uint16'`` to ``(full_scale, torch.dtype)``."""
    if name not in _UINT_INFO:
        raise InputContractError(
            f"Unknown dtype {name!r}; allowed: {list(_UINT_INFO)}"
        )
    return _UINT_INFO[name]


def _apply_image2image(
    logits: torch.Tensor, *, ch_axis: int, dtype: str,
) -> torch.Tensor:
    """image2image output transform.

    ``logits`` → ``sigmoid → clamp[0,1] → ·(2ⁿ-1) → round → dtype``, then squeeze
    the C=1 channel axis (the single-channel collapse).

    The caller guarantees ``logits.shape[ch_axis] == 1`` (config.out_channels=1
    for image2image; validated at the public call site).
    """
    full_scale, torch_dtype = _resolve_dtype(dtype)
    activated = torch.sigmoid(logits)
    clamped = activated.clamp(min=0.0, max=1.0)
    scaled = clamped * full_scale
    rounded = scaled.round().to(torch_dtype)
    return rounded.squeeze(ch_axis)


def _apply_image2label(
    logits: torch.Tensor, *, ch_axis: int, dtype: str,
) -> torch.Tensor:
    """image2label output transform — argmax over the channel axis.

    The caller guarantees ``logits.shape[ch_axis] >= 2`` (config.out_channels>=2
    for image2label; validated at the public call site).
    Returns the class map cast to ``dtype``.
    """
    _, torch_dtype = _resolve_dtype(dtype)
    return logits.argmax(dim=ch_axis).to(torch_dtype)


__all__ = ["LabelDtype"]

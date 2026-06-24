"""ModelConfig — the user-owned recipe to build an OmniEM.

A ``ModelConfig`` is the user-owned recipe to build an :class:`omniem.models.OmniEM`
and to validate inputs at :meth:`OmniEM.predict` time.

The model returns pure logits; the activation is a property of ``task_type``,
applied by :meth:`OmniEM.run` (its internal output stage), not the model forward
(there is no ``output_nonlinear`` field).

* **`task_type`** (optional, ``Literal['image2image','image2label'] | None``).
  When set, it derives/validates the output side: ``image2image`` derives
  ``out_channels=1`` (must equal 1 if given); ``image2label`` requires
  ``out_channels>=2``. When ``None``, the model can't decide the postprocess and
  the caller writes raw logits.
* **`out_channels`** is optional (`int | None`): *derived* for ``image2image``
  via a ``model_validator(mode="before")`` (so the user can omit it), then
  *enforced* by the after-validator. We avoid mutating ``self`` in an
  after-validator because :class:`~omniem.config.BaseConfig` sets
  ``validate_assignment=True`` and mutation would re-trigger validation.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from omniem.config.base import BaseConfig
from omniem.errors import OmniEMWarning

TaskType = Literal["image2image", "image2label"]


class ModelConfig(BaseConfig):
    """Model configuration.

    Fields:
        schema_version: Inherited from :class:`~omniem.config.BaseConfig`.
        arch: The :data:`MODEL_ARCH_REGISTRY` key (e.g. ``"omniemv1"``) — selects
            the factory that owns the pinned constants. The only omniem-side
            field; everything else mirrors the model constructor.
        encoder: The :data:`~omniem.encoders.registry.ARCH_REGISTRY` name of the
            ViT backbone (e.g. ``"emdinov1"``).
        img_z: Exact Z size of one input unit. ``1`` → 2D model (no z-fusion);
            ``>1`` → 3D model (z-fusion via ``kernel3d_z``). Replaces the
            redundant ``unit`` field.
        out_channels: Decoder output channels. Optional —
            ``image2image`` derives it to ``1`` if omitted; ``image2label``
            requires it (``>=2``); ``task_type=None`` requires it explicitly.
        kernel3d_z: Z-kernel for the decoder downsample convs. ``img_z==1`` →
            must be ``None``. ``img_z>1`` → ``None`` (treated as 1) or a
            positive ODD int (3 for the mito-3D head).
        task_type: ``"image2image"`` (denoise/super-res →
            sigmoid+uint output via :meth:`OmniEM.run`), ``"image2label"``
            (segmentation → argmax output), or ``None`` (model has no opinion;
            the caller postprocesses the raw logits themselves). The activation
            is **derived** from this — there is no `output_nonlinear` config field.
        resize4emdino: Bicubic-resize the input from the omniem-patch grid to
            the ViT-patch grid. Default ``False``. Per-head.
        mean: Fixed training-norm subtrahend, expressed in the ``[0, 1]`` range
            (the caller divides ints by 255 / etc. before predict). This is the
            head's own training mean — NOT per-image, NOT an arch default.
        std: Fixed training-norm divisor (the head's own training std). Must be
            finite and ``> 0``.
    """

    arch: str
    encoder: str
    img_z: int = Field(gt=0)
    out_channels: int | None = None
    kernel3d_z: int | None = None
    task_type: TaskType | None = None
    resize4emdino: bool = False
    mean: float
    std: float

    # ---- field validators --------------------------------------------------------

    @field_validator("mean")
    @classmethod
    def _validate_mean(cls, v: float) -> float:
        """Finite float — NaN/inf is never a valid training mean.

        WARN (never raise) when ``mean`` falls outside ``[0, 1]``: omniem model
        configs use ``[0, 1]``-domain stats (the input is scaled to ``[0, 1]``
        before normalization). A value like ``136`` is almost certainly a raw
        ``0-255`` stat that should be divided by 255 (`OmniEMWarning`).
        """
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"mean must be a float (got {type(v).__name__})")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"mean must be finite (got {f})")
        if not (0.0 <= f <= 1.0):
            warnings.warn(
                f"ModelConfig.mean={f} is outside [0,1]; omniem normalizes a "
                f"[0,1]-scaled input. If this is a raw 0-255 stat, divide by 255 "
                f"(e.g. {f}/255 = {f / 255.0:.6f}).",
                category=OmniEMWarning,
                stacklevel=2,
            )
        return f

    @field_validator("std")
    @classmethod
    def _validate_std(cls, v: float) -> float:
        """Finite + strictly positive — std=0 would divide-by-zero in the affine.

        WARN (never raise) when ``std`` falls outside ``(0, 1]`` — same ÷255
        migration hint as ``mean`` (`OmniEMWarning`).
        """
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"std must be a float (got {type(v).__name__})")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"std must be finite (got {f})")
        if f <= 0.0:
            raise ValueError(f"std must be > 0 (got {f}); std divides the input")
        if f > 1.0:
            warnings.warn(
                f"ModelConfig.std={f} is outside (0,1]; omniem normalizes a "
                f"[0,1]-scaled input. If this is a raw 0-255 stat, divide by 255 "
                f"(e.g. {f}/255 = {f / 255.0:.6f}).",
                category=OmniEMWarning,
                stacklevel=2,
            )
        return f

    # ---- task_type → out_channels derivation (before validator) ------------------
    #
    # We FILL out_channels=1 here when the user picked task_type='image2image' and
    # omitted out_channels. Using `mode="before"` so we mutate the input dict, NOT
    # a constructed model instance — BaseConfig sets validate_assignment=True (see
    # omniem/config/base.py:103), so mutating in an after-validator would re-trigger
    # the assignment validator and is brittle / can recurse.

    @model_validator(mode="before")
    @classmethod
    def _fill_image2image_out_channels(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("task_type") == "image2image" and data.get("out_channels") is None:
            data = {**data, "out_channels": 1}
        return data

    # ---- cross-field validation (after) ------------------------------------------

    @model_validator(mode="after")
    def _validate_shape_rules(self) -> ModelConfig:
        """Cross-field rules — ``kernel3d_z`` shape, ``task_type`` ↔ ``out_channels``.

        ``kernel3d_z`` rules:
        - ``img_z == 1``: ``kernel3d_z`` MUST be ``None`` (no z-fusion possible).
        - ``img_z > 1``: ``kernel3d_z`` is ``None`` (treated as 1) or a
          positive ODD int (3 for mito-3D).

        ``task_type`` ↔ ``out_channels`` rules:
        - ``"image2image"``: ``out_channels`` must equal ``1`` (or omit; the
          before-validator filled it).
        - ``"image2label"``: ``out_channels`` is required and must be ``>= 2``.
        - ``None``: ``out_channels`` is required (the decoder needs it).
        """
        # kernel3d_z rules.
        if self.img_z == 1:
            if self.kernel3d_z is not None:
                raise ValueError(
                    f"img_z==1 requires kernel3d_z=None "
                    f"(no z-fusion possible); got kernel3d_z={self.kernel3d_z}"
                )
        else:
            # img_z > 1 — 3D model.
            if self.kernel3d_z is not None:
                k = self.kernel3d_z
                if not isinstance(k, int) or isinstance(k, bool):
                    raise ValueError(f"kernel3d_z must be int|None (got {type(k).__name__})")
                if k <= 0:
                    raise ValueError(f"kernel3d_z must be > 0 when set (got {k})")
                if k % 2 != 1:
                    raise ValueError(
                        f"kernel3d_z must be odd (positive odd int; got {k})"
                    )

        # task_type ↔ out_channels rules.
        if self.task_type == "image2image":
            if self.out_channels is None:
                # The before-validator should have filled this; this guards a
                # direct ModelConfig.model_construct or pathological path.
                raise ValueError(
                    "task_type='image2image' requires out_channels==1 "
                    "(omit it to derive the default)"
                )
            if self.out_channels != 1:
                raise ValueError(
                    f"task_type='image2image' fixes out_channels to 1 "
                    f"(got out_channels={self.out_channels}); omit the field or set it to 1"
                )
        elif self.task_type == "image2label":
            if self.out_channels is None:
                raise ValueError(
                    "task_type='image2label' requires an explicit out_channels "
                    "(>= 2: background + classes)"
                )
            if self.out_channels < 2:
                raise ValueError(
                    f"task_type='image2label' requires out_channels >= 2 "
                    f"(background + at least one class); got out_channels={self.out_channels}"
                )
        else:
            # task_type is None — the user owns postprocess; we still need
            # out_channels to build the decoder.
            if self.out_channels is None:
                raise ValueError(
                    "out_channels is required when task_type is None "
                    "(the decoder needs it; set task_type to derive image2image "
                    "or pass an explicit count for image2label)"
                )
            if self.out_channels <= 0:
                raise ValueError(
                    f"out_channels must be > 0 (got {self.out_channels})"
                )
        return self


__all__ = ["ModelConfig", "TaskType"]

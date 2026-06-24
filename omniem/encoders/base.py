"""EMEncoder (nn.Module) — the omniem public encoder wrapper.

The backbone is held under the **arch-owned** attribute name
(``ArchInfo.backbone_attr`` — ``"vit"`` for ViT archs, so :meth:`EMEncoder.state_dict`
keys are ``vit.*``, identical to the raw checkpoint layout). The general wrapper
reads that name from the arch instead of hard-coding ``vit``; weights load via a
plain ``torch.load(weights_only=True)`` + ``self.load_state_dict(strict=True)``
— **no key stripping, no key renaming**. Internal code addresses the backbone via
``self._backbone`` (arch-agnostic); for ViT archs that *is* ``self.vit``.

An encoder is fully specified by **(arch name, weights file)** — there is no
encoder config object and no tag. The arch name selects the architecture + its
frozen pretraining normalisation (``mean``/``std``, from ``ARCH_REGISTRY``); the
weights file is the specific pretrained checkpoint (a raw ``vit.*``
checkpoint, loaded directly — no re-export).

Usage:

    enc = EMEncoder.load("emdinov1", "weights/emdino_v3_best_250703.pth")
    out = enc.run(x, axes="yx")                   # -> {"cls": [1, 1, 1024]}

Public surface (0.1.1 — channel-less, two-tier):

    EMEncoder.load(arch, weights, *, device=None, dtype=None)
    enc.run(image, *, axes, norm=None, conform="strict", squeeze="",
            return_cls=True, return_patch=False, return_blocks=None,
            block_callback=None)                  # raw image -> feature dict
    enc.forward(tensor, *, return_cls=True, return_patch=False,
                return_blocks=None, block_callback=None, squeeze="")
                                                  # canonical [b, z, y, x] -> dict
    enc.arch             # str
    enc.mean, enc.std    # float — the arch's pretraining normalisation
    enc.device           # torch.device
    enc.dtype            # torch.dtype
    enc.embed_dim        # int

``image`` / ``tensor`` must be **grayscale** (gray0) — ``axes`` may not contain
``c``; the encoder synthesises the backbone's channel layout itself. The canonical
compute layout is ``[b, z, y, x]`` (``z=1`` for a 2D tile). See
``docs/input-format.md``.

The ``norm=`` argument takes one of four shapes:
``None`` → use the arch ``mean`` / ``std``; ``'per-image'`` → per-sample z-score
``(x − mean(x)) / std(x)`` (scale-invariant); ``'prenormalized'`` → skip the affine
(the caller already normalised); ``{'mean': m, 'std': s}`` → override (both keys
required). For custom maths, normalise the tensor yourself and pass
``norm='prenormalized'``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from omniem.encoders.dinov2.build import build
from omniem.encoders.dinov2.forward import (
    compute_encoder,
    prepare_encoder_input,
)
from omniem.encoders.registry import arch_info, arch_mean_std
from omniem.errors import InputContractError, WeightFormatError

# Public ``ArrayLike`` — the input to ``EMEncoder.forward`` / ``OmniEM.predict``.
# Float arrays only; integer arrays must be scaled by the caller (uint8 != ÷255).
ArrayLike = torch.Tensor | np.ndarray


class EMEncoder(nn.Module):
    """omniem encoder wrapper. See module docstring."""

    def __init__(self, backbone: nn.Module, arch: str) -> None:
        super().__init__()
        info = arch_info(arch)
        # The backbone attribute name is ARCH-OWNED (info.backbone_attr), not
        # hard-coded here — the general wrapper reads ``vit`` (etc.) from the arch.
        # It drives EMEncoder.state_dict() keys (e.g. ``vit.*``), identical to the
        # raw checkpoint layout, so weights load via strict=True without stripping.
        # Access the backbone internally via ``self._backbone`` (arch-agnostic);
        # ``self.vit`` still resolves for ViT archs because that IS the attr name.
        self._backbone_attr = info.backbone_attr
        setattr(self, self._backbone_attr, backbone)
        self._arch = arch
        # The arch carries its frozen pretraining normalisation (ARCH_REGISTRY).
        # That is the encoder's effective mean/std unless a forward-time ``norm=``
        # override is passed.
        self._mean, self._std = arch_mean_std(arch)
        # Runtime anti-drift cross-check: the arch catalog's
        # ``stride`` MUST equal the built backbone's ``patch_size``. The static
        # registry-equality test already catches registry-side typos, but a
        # future factory edit that changed patch_size without updating the
        # catalog would slip past it. Catching the drift here surfaces it at
        # construction time rather than silently mis-padding/resizing later.
        catalog_stride = int(info.stride)
        built_patch = int(backbone.patch_size)
        if catalog_stride != built_patch:
            raise InputContractError(
                f"arch_info({arch!r}).stride={catalog_stride} disagrees with the "
                f"built backbone's patch_size={built_patch}. The arch catalog is "
                f"the documented source of truth for the encoder input divisor "
                f"— update one of them."
            )
        # Inference-only by contract — :meth:`load` flips this; the bare ctor (used
        # by the lightweight test suite) calls eval() here too so it is safe.
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    # ---- introspection -------------------------------------------------------------

    @property
    def _backbone(self) -> nn.Module:
        """The bare backbone module, addressed by the arch-owned attribute name.

        Equivalent to ``self.vit`` for ViT archs, but arch-agnostic — internal code
        and the model-layer borrow path use this instead of hard-coding ``vit``.
        """
        return getattr(self, self._backbone_attr)

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def mean(self) -> float:
        """The arch's pretraining normalisation mean (the effective default)."""
        return self._mean

    @property
    def std(self) -> float:
        """The arch's pretraining normalisation std (the effective default)."""
        return self._std

    @property
    def device(self) -> torch.device:
        """The device the parameters live on. Reads one parameter (no overhead)."""
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover — paramless backbone is impossible
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:  # pragma: no cover
            return torch.float32

    @property
    def embed_dim(self) -> int:
        return int(self._backbone.embed_dim)

    @property
    def n_blocks(self) -> int:
        return int(self._backbone.n_blocks)

    def name_parameter_group(self) -> str:
        """Return the top-level name shared by the encoder's parameters.

        For ViT archs (``emdinov1`` / the tiny test arch) the backbone is held at
        ``self.vit`` (the arch's ``backbone_attr``), so ``named_parameters()`` keys
        are ``vit.blocks.…`` / ``vit.patch_embed.…`` — this returns ``"vit"``.

        Derived from the live ``named_parameters()`` (the first component of the
        first parameter name) rather than hard-coded, so an arch that declares a
        different ``backbone_attr`` reports that name instead. Useful for callers (e.g. ``omniem-train``) that need to address
        the encoder's parameters as a group — to set a distinct learning rate,
        freeze them, or partition an optimiser's ``param_groups``.

        Returns:
            The shared leading dotted-name component (e.g. ``"vit"``).

        Raises:
            InputContractError: The encoder has no parameters, OR its parameters
                do not all share a single top-level group (no unambiguous name).
        """
        groups = {name.split(".", 1)[0] for name, _ in self.named_parameters()}
        if not groups:
            raise InputContractError(
                "EMEncoder.name_parameter_group: the encoder has no parameters."
            )
        if len(groups) != 1:
            raise InputContractError(
                f"EMEncoder.name_parameter_group: parameters span multiple "
                f"top-level groups {sorted(groups)} — no single group name. "
                f"(EMEncoder is expected to hold its backbone under one "
                f"attribute, e.g. self.vit.)"
            )
        return next(iter(groups))

    # ---- run (merged one-step: prep → compute) -----------------------------------

    def run(
        self,
        image: ArrayLike,
        *,
        axes: str,
        norm: None | str | Mapping[str, float] = None,
        conform: str = "strict",
        squeeze: str = "",
        return_cls: bool = True,
        return_patch: bool = False,
        return_blocks: Sequence[int] | None = None,
        block_callback: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Run the full encoder pipeline from a raw image — the everyday path.

        ``run`` = ``_apply_input → forward`` with the prep stage threaded
        internally, so a mismatch is impossible. The input is **channel-less**
        grayscale; ``axes`` may not contain ``c`` (the encoder synthesises the
        backbone's channels itself).

        Args:
            image: Raw grayscale input — a float :class:`torch.Tensor` or
                :class:`numpy.ndarray` (integer dtypes rejected; the caller scales
                int→float).
            axes: One-character-per-axis string from ``{b, z, y, x}`` describing
                ``image`` (whitespace ignored; a ``c`` axis raises).
            norm: ``None`` → arch ``mean``/``std``; ``'per-image'`` → per-sample
                z-score; ``'prenormalized'`` → skip the affine; ``{'mean': m,
                'std': s}`` → **scalar** override (per-channel sequences rejected).
            conform: ``'strict'`` (default — reject non-square /
                non-stride-multiple XY) or ``'resize'`` (bicubic XY-only resize to
                the next square multiple of the patch stride). ``'pad'`` is rejected.
            squeeze: A subset of ``{b, z}`` — drop the named axis from the feature
                output **iff** singleton (else raise). Default ``""`` keeps the full
                ``[B, Z, …]`` shape.
            return_cls / return_patch / return_blocks / block_callback: forwarded
                to :meth:`forward` (the compute stage).

        Returns:
            The feature dict (``cls`` / ``patch`` / ``inner``). Encoder features
            have no spatial XY to mirror, so the shape follows ``[B, Z, …]`` (after
            ``squeeze``) regardless of ``axes``.
        """
        tensor, _orig_yx = self._apply_input(image, axes=axes, norm=norm, conform=conform)
        return self.forward(
            tensor,
            return_cls=return_cls,
            return_patch=return_patch,
            return_blocks=return_blocks,
            block_callback=block_callback,
            squeeze=squeeze,
        )

    # ---- internal prep stage ------------------------------------------------------

    def _apply_input(
        self,
        image: ArrayLike,
        *,
        axes: str,
        norm: None | str | Mapping[str, float] = None,
        conform: str = "strict",
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Internal prep stage — return ``(canonical[b, z, y, x], orig_yx)``.

        Owns the wrapper-level validation (fp16-CPU, int-dtype, ndarray→tensor,
        device/dtype move) then delegates to
        :func:`omniem.encoders.dinov2.forward.prepare_encoder_input`. The returned
        tensor is channel-less and normalised; channel synthesis + ``B*Z`` fold are
        compute concerns (:meth:`forward`). Not public — :meth:`run` threads it.
        """
        if self.dtype == torch.float16 and self.device.type == "cpu":
            raise InputContractError(
                "Encoder is float16 on CPU — most CPU conv kernels do not support fp16. "
                "Move the encoder to a CUDA device, or use float32 on CPU."
            )
        image = self._coerce_input(image)
        return prepare_encoder_input(
            self._backbone,
            image,
            axes=axes,
            mean=self._mean,
            std=self._std,
            norm=norm,
            conform=conform,
        )

    # ---- forward (canonical compute) ---------------------------------------------

    def forward(  # type: ignore[override]
        self,
        tensor: torch.Tensor,
        *,
        return_cls: bool = True,
        return_patch: bool = False,
        return_blocks: Sequence[int] | None = None,
        block_callback: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        squeeze: str = "",
        axes: str | None = None,
        norm: Any = None,
        conform: Any = None,
    ) -> dict[str, Any]:
        """Compute encoder features from a **canonical** ``[b, z, y, x]`` tensor.

        The power path: ``tensor`` is a pre-built canonical, channel-less,
        **already-normalised** float tensor (``z=1`` for a 2D tile). ``forward``
        validates the shape strictly (no normalization inferred), folds ``B*Z``,
        synthesises the channel axis, runs the blocks, un-folds to ``[B, Z, …]``,
        and applies ``squeeze``. For raw images use :meth:`run`.

        Args:
            tensor: Canonical ``[b, z, y, x]`` float tensor — integer tensors raise
                (the package never scales int→float). Auto-moved/cast to the
                encoder's device/dtype.
            return_cls / return_patch / return_blocks / block_callback: the
                return-flag contract (independent bools; ``return_blocks=[i, …]``
                both requests and selects the taps; all-false raises). The
                ``block_callback`` sees the folded ``[B*Z, tokens, dim]``.
            squeeze: A subset of ``{b, z}`` — drop the named singleton axis.
            axes / norm / conform: **Removed in 0.1.1.** Present only to detect the
                old ``forward(raw, axes=…, norm=…, conform=…)`` one-shot call and
                raise a clear migration error rather than a cryptic ``TypeError``.

        Returns:
            ``{cls [B, Z, D]?, patch [B, Z, N, D]?, inner {i: [B, Z, T, D]}?}``
            (after ``squeeze``).
        """
        if axes is not None or norm is not None or conform is not None:
            raise InputContractError(
                "EMEncoder.forward(x, axes=…/norm=…/conform=…) was removed in 0.1.1. "
                "`forward` now takes only a canonical [b, z, y, x] tensor. Use "
                "`enc.run(image, axes=…, norm=…, conform=…)` for a raw image, or build "
                "the canonical tensor and call `enc.forward(canonical)`."
            )
        if self.dtype == torch.float16 and self.device.type == "cpu":
            raise InputContractError(
                "Encoder is float16 on CPU — most CPU conv kernels do not support fp16. "
                "Move the encoder to a CUDA device, or use float32 on CPU."
            )
        if not isinstance(tensor, torch.Tensor):
            raise InputContractError(
                f"EMEncoder.forward expects a canonical [b, z, y, x] torch.Tensor "
                f"(got {type(tensor).__name__}); use run(image, axes=…) for raw input."
            )
        # Auto-move/cast the float tensor onto the encoder (compute validates int).
        if tensor.is_floating_point() and (
            tensor.device != self.device or tensor.dtype != self.dtype
        ):
            tensor = tensor.to(device=self.device, dtype=self.dtype)
        return compute_encoder(
            self._backbone,
            tensor,
            return_cls=return_cls,
            return_patch=return_patch,
            return_blocks=return_blocks,
            block_callback=block_callback,
            squeeze=squeeze,
        )

    # ---- migration stub (removed public surface) ---------------------------------

    def apply_input(self, *args: Any, **kwargs: Any):
        """Removed in 0.1.1 — use :meth:`run` / build a canonical for :meth:`forward`."""
        raise InputContractError(
            "EMEncoder.apply_input was removed in 0.1.1 (no more Prepared carrier). "
            "Use `enc.run(image, axes=…)` for the full pipeline, or build a "
            "canonical [b, z, y, x] tensor and call `enc.forward(canonical)`."
        )

    # ---- wrapper-level coercion (owned by _apply_input) --------------------------

    def _coerce_input(self, x: ArrayLike) -> torch.Tensor:
        """Reject int dtypes / wrong types and move to self.device/self.dtype.

        Shared by :meth:`_apply_input`; the prep owns its validation.
        """
        if isinstance(x, np.ndarray):
            if not np.issubdtype(x.dtype, np.floating):
                raise InputContractError(
                    f"EMEncoder requires a float ndarray "
                    f"(got dtype={x.dtype}); the package does not guess int->float "
                    f"scaling. Use x.astype(np.float32) / SCALE on the caller side."
                )
            x = torch.from_numpy(np.ascontiguousarray(x))
        elif isinstance(x, torch.Tensor):
            if not torch.is_floating_point(x):
                raise InputContractError(
                    f"EMEncoder requires a float torch.Tensor "
                    f"(got dtype={x.dtype}); the package does not guess int->float "
                    f"scaling (uint8 != ÷255). Use x.float() / SCALE on the caller "
                    f"side first."
                )
        else:
            raise InputContractError(
                f"EMEncoder x must be a torch.Tensor or numpy.ndarray "
                f"(got {type(x).__name__})"
            )
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(device=self.device, dtype=self.dtype)
        return x

    # ---- load ----------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        arch: str,
        weights: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> EMEncoder:
        """Build the ``arch`` backbone and strict-load the ``vit.*`` ``weights``.

        The raw ``vit.*`` checkpoint is loaded with a plain
        ``torch.load(weights_only=True)`` + ``self.load_state_dict(strict=True)``
        — NO key stripping, NO key renaming. ``EMEncoder.state_dict()`` keys are
        ``vit.*`` (because the backbone is held as ``self.vit``).

        Args:
            arch: The :data:`omniem.encoders.registry.ARCH_REGISTRY` key (e.g.
                ``"emdinov1"``). Selects the architecture + its frozen pretraining
                normalisation. An unknown arch raises :class:`ConfigError`.
            weights: Path to the ``vit.*`` checkpoint (e.g.
                ``weights/backbone_emdino_v1.pt``).
            device: Target device (default CPU).
            dtype: Target dtype (default the backbone's native dtype).

        Returns:
            A ready-to-use :class:`EMEncoder` in eval mode with grads off.

        Raises:
            ConfigError: unknown ``arch``.
            WeightFormatError: unreadable / structurally-wrong weights, or a strict
                ``load_state_dict`` mismatch.
        """
        backbone = build(arch)  # raises ConfigError on unknown arch
        enc = cls(backbone, arch)

        path = Path(weights)
        try:
            state_dict = torch.load(path, weights_only=True, map_location="cpu")
        except FileNotFoundError:
            raise
        except Exception as e:
            raise WeightFormatError(
                f"{path}: cannot read weights file "
                f"(corrupt, not a torch checkpoint, or wrong format): {e}"
            ) from e

        if not isinstance(state_dict, dict):
            raise WeightFormatError(
                f"{path}: weights must be a state_dict (got {type(state_dict).__name__})"
            )

        try:
            result = enc.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise WeightFormatError(
                f"{path}: strict load_state_dict failed: {e}"
            ) from e
        # Defensive — strict=True already raises on any mismatch, but re-check for
        # alternate torch versions that surface mismatches via the result object.
        if result.missing_keys or result.unexpected_keys:
            raise WeightFormatError(
                f"{path}: strict load_state_dict left unmatched keys "
                f"(missing={result.missing_keys[:5]}, "
                f"unexpected={result.unexpected_keys[:5]})"
            )

        target_device = torch.device(device) if device is not None else torch.device("cpu")
        target_dtype = dtype if dtype is not None else next(backbone.parameters()).dtype
        # Guard fp16-on-CPU at load time so the error surfaces here, not at forward.
        if target_dtype == torch.float16 and target_device.type == "cpu":
            raise InputContractError(
                "Cannot load encoder as float16 on CPU — most CPU conv kernels do "
                "not support fp16. Use device='cuda' or dtype=torch.float32."
            )
        enc.to(device=target_device, dtype=target_dtype)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        return enc


__all__ = ["EMEncoder"]

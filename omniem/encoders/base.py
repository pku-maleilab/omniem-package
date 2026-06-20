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
    out = enc(x, axes="yx")                       # -> {"cls": [1, 1024]}

Public surface:

    EMEncoder.load(arch, weights, *, device=None, dtype=None)
    enc.forward(x, *, axes, return_cls=True, return_patch=False,
                return_blocks=None, block_callback=None, norm=None)
    enc.arch             # str
    enc.mean, enc.std    # float — the arch's pretraining normalisation
    enc.device           # torch.device
    enc.dtype            # torch.dtype
    enc.embed_dim        # int

``x`` must be **grayscale** (gray0) — ``axes`` may not contain ``c``; the encoder
synthesises the backbone's channel layout itself. See ``docs/input-format.md``.

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
    apply_input_features,
    compute_features,
)
from omniem.encoders.registry import arch_info, arch_mean_std
from omniem.errors import InputContractError, WeightFormatError
from omniem.prepared import Prepared

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

    # ---- forward -------------------------------------------------------------------

    # ---- apply_input --------------------------------------------------------------

    def apply_input(
        self,
        x: ArrayLike,
        *,
        axes: str,
        norm: None | str | Mapping[str, float | Sequence[float]] = None,
        conform: str = "resize",
    ) -> Prepared:
        """Run only the input-transform stage; return a :class:`Prepared` carrier.

        The split-out front half of :meth:`forward` — axes fold + gray0 →
        ``in_chans`` synthesis + mean/std affine + XY conform. The wrapper-level
        validation (fp16-CPU, int-dtype, ndarray→tensor) is owned by this method;
        :meth:`forward` delegates here on raw input. Unlike the model-side
        :meth:`OmniEM.apply_input`, this moves the input onto
        ``self.device``/``self.dtype`` (a CUDA encoder yields a CUDA
        ``Prepared.tensor``).

        Args:
            x: Raw grayscale input — a float :class:`torch.Tensor` or
                :class:`numpy.ndarray`. ``axes`` may not contain ``c`` (the
                encoder synthesises the backbone's channels). Integer dtypes are
                rejected.
            axes: One-character-per-axis string from ``{b,z,y,x}`` describing
                ``x``.
            norm: ``None`` → arch ``mean``/``std``; ``'per-image'`` → per-sample
                z-score; ``'prenormalized'`` → skip the affine; ``{'mean': m,
                'std': s}`` → override. Out-of-``[0,1]`` mean/std or scaled input →
                warn-only :class:`~omniem.errors.OmniEMWarning`.
            conform: ``'resize'`` (default — bicubic XY-only interpolate to the
                next square multiple of the encoder's patch stride) or
                ``'strict'`` (reject non-square / non-stride-multiple XY).
                ``'pad'`` is not supported here.

        Returns:
            A frozen :class:`~omniem.prepared.Prepared` carrying the canonical
            ``[B*Z, in_chans, S, S]`` tensor plus the metadata downstream
            consumers need (``axes``, ``conform``, ``orig_yx``, ``pad_or_scale``,
            ``B``, ``Z``, ``stride``).

        Raises:
            InputContractError: fp16-on-CPU; a non-float / wrong-type input; or an
                axes string that does not match ``x``.
        """
        # fp16 CPU guard — apply_input owns the wrapper
        # validation; the standalone-encoder split path otherwise
        # surfaces a kernel error inside the backbone.
        if self.dtype == torch.float16 and self.device.type == "cpu":
            raise InputContractError(
                "Encoder is float16 on CPU — most CPU conv kernels do not support fp16. "
                "Move the encoder to a CUDA device, or use float32 on CPU."
            )
        x = self._coerce_input(x)
        return apply_input_features(
            self._backbone,
            x,
            axes=axes,
            mean=self._mean,
            std=self._std,
            norm=norm,
            conform=conform,
        )

    # ---- forward -------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        x: ArrayLike | Prepared,
        *,
        axes: str | None = None,
        return_cls: bool = True,
        return_patch: bool = False,
        return_blocks: Sequence[int] | None = None,
        block_callback: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        norm: None | str | Mapping[str, float | Sequence[float]] = None,
        conform: str = "strict",
    ) -> dict[str, Any]:
        """Run the omniem-driven forward pass.

        ``x`` may be a raw :class:`torch.Tensor`/ndarray OR a
        :class:`Prepared` returned by :meth:`apply_input`. The type is the
        signal — no ``prepared=`` flag. When ``x`` is a ``Prepared``,
        ``axes`` / ``norm`` / ``conform`` MUST be omitted / left at defaults
        (they are baked into the meta) — passing non-defaults raises.

        ``conform`` defaults to ``'strict'`` here (preserves the reject-on-non-
        conforming-XY behaviour on
        raw input — a non-conforming XY raises). :meth:`apply_input` defaults to
        ``'resize'`` instead because callers that reach for the split path are
        the ones who want the round-trip.

        Raw-input contract (unchanged):
        * ``x`` may be a :class:`torch.Tensor` OR a :class:`numpy.ndarray`.
        * Integer inputs are rejected (uint8 != ÷255).
        * ``x`` must be **grayscale** (gray0) — ``axes`` may not contain ``c``.
        * ``axes`` is required; ``norm`` / ``conform`` follow
          :meth:`apply_input`.

        Return-flag contract: ``return_cls`` / ``return_patch`` are independent
        booleans; ``return_blocks=[i, j, …]`` both requests and selects the inner
        taps. All-false → :class:`InputContractError`.

        Device/dtype: input auto-moved to ``self.device`` + auto-cast to
        ``self.dtype``. No silent half on CPU.
        """
        if self.dtype == torch.float16 and self.device.type == "cpu":
            raise InputContractError(
                "Encoder is float16 on CPU — most CPU conv kernels do not support fp16. "
                "Move the encoder to a CUDA device, or use float32 on CPU."
            )

        if isinstance(x, Prepared):
            # Prepared-mode: axes / norm / conform are baked in — reject overrides.
            if axes is not None:
                raise InputContractError(
                    "EMEncoder.forward: `axes` must be omitted when x is a Prepared "
                    "(axes is already baked into Prepared.axes)."
                )
            if norm is not None:
                raise InputContractError(
                    "EMEncoder.forward: `norm` must be omitted when x is a Prepared "
                    "(the affine was already applied during apply_input)."
                )
            if conform != "strict":
                # ``conform`` has a default, but a caller passing a non-default
                # value alongside a Prepared is almost certainly confused — flag
                # it so they don't think it changes anything. (Default is
                # ``'strict'`` here for raw-input back-compat; the Prepared
                # already carries its own ``conform`` in the meta.)
                raise InputContractError(
                    "EMEncoder.forward: `conform` must be omitted when x is a Prepared "
                    "(the conform was already applied during apply_input)."
                )
            # Coerce the Prepared tensor onto the encoder's device/dtype before
            # the backbone forward. Without this a CPU/double
            # Prepared fed to a CUDA/float32 encoder would surface a baffling
            # internal torch error instead of the wrapper's "auto-moved" promise.
            if x.tensor.device != self.device or x.tensor.dtype != self.dtype:
                # Prepared is frozen, so rebuild it with the moved tensor.
                x = Prepared(
                    tensor=x.tensor.to(device=self.device, dtype=self.dtype),
                    axes=x.axes,
                    conform=x.conform,
                    orig_yx=x.orig_yx,
                    pad_or_scale=x.pad_or_scale,
                    B=x.B,
                    Z=x.Z,
                    stride=x.stride,
                )
            return compute_features(
                self._backbone,
                x,
                return_cls=return_cls,
                return_patch=return_patch,
                return_blocks=return_blocks,
                block_callback=block_callback,
            )

        # Raw-input path: axes is required.
        if axes is None:
            raise InputContractError(
                "EMEncoder.forward: `axes` is required for raw input (pass a "
                "Prepared via x= to skip; see apply_input)."
            )
        x = self._coerce_input(x)
        prepared = apply_input_features(
            self._backbone,
            x,
            axes=axes,
            mean=self._mean,
            std=self._std,
            norm=norm,
            conform=conform,
        )
        return compute_features(
            self._backbone,
            prepared,
            return_cls=return_cls,
            return_patch=return_patch,
            return_blocks=return_blocks,
            block_callback=block_callback,
        )

    # ---- wrapper-level coercion (owned by apply_input + forward) ------------------

    def _coerce_input(self, x: ArrayLike) -> torch.Tensor:
        """Reject int dtypes / wrong types and move to self.device/self.dtype.

        Pulled out of the old ``forward`` body so :meth:`apply_input` and the
        raw-input branch of :meth:`forward` share exactly one validator
        (the prep owns its validation; nothing is left behind when the split
        happens).
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

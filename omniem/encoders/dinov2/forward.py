"""omniem-driven forward pass for the DINOv2 backbone (channel-less, two-stage).

This module reimplements the backbone's ``forward_features`` loop so it can host
the omniem block hook (``block_callback``) and the public ``inner=`` taps. The
underlying vendored class is **not** edited.

The 0.1.1 *unify-io* redesign split the pass into two explicit stages, both
**channel-less** (EM is grayscale — the caller never declares a ``c`` axis):

* :func:`prepare_encoder_input` — the prep stage. Validates ``axes`` (``c``
  rejected), reorders the caller's input into the canonical ``[b, z, y, x]``
  layout (``z`` explicit, ``z=1`` for a 2D tile), captures the pre-conform XY as
  ``orig_yx``, conforms XY (``strict`` / ``resize``; ``pad`` is rejected here),
  and applies the **scalar** ``(x − mean) / std`` affine. It returns a plain
  ``(tensor, orig_yx)`` — **no** ``Prepared`` carrier, **no** channel synthesis,
  **no** ``B*Z`` fold (those are compute concerns now).
* :func:`compute_encoder` — the compute stage. Strict-validates the canonical
  ``[b, z, y, x]`` tensor (float, rank, square + patch-aligned XY), folds
  ``(B, Z) → B*Z`` so the blocks stay strictly 2D, synthesises the channel axis
  (gray → ``in_chans`` repeat), drives the block loop (``block_callback`` sees the
  folded ``[B*Z, tokens, dim]``), then **un-folds** every output back to explicit
  ``[B, Z, …]`` and applies the ``squeeze`` directive (drop singleton ``b`` / ``z``).

The return shape is a flat dict — adapters / heads are NOT part of this signature
(they live in head-owned wrappers).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from omniem._pipeline import parse_squeeze
from omniem.errors import InputContractError, OmniEMWarning

# ``norm=`` accepts one of four shapes (replacing the old
# mean=/std=/transform= trio):
#   * None            → use the arch ``mean`` / ``std`` (passed in);
#   * 'prenormalized' → skip the affine (the caller already did it);
#   * 'per-image'     → per-sample z-score (x - mean(x)) / std(x);
#   * {'mean', 'std'} → SCALAR override (both keys required; sequences rejected).
# The literal strings the caller passes.
_PRENORMALIZED = "prenormalized"
_PER_IMAGE = "per-image"

# Constant-image guard for per-image z-score: std below this → divide is skipped
# (subtract mean only, yielding zeros for a flat image — no blow-up).
_PER_IMAGE_STD_EPS = 1e-8

# Tolerance for the [0,1] range warn so bicubic-resize overshoot / float noise
# does not trip a spurious warning (a genuinely wrong domain is off by ≫ this).
_UNIT_RANGE_TOL = 1e-3


def _per_image_stats(x_preconform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample mean/std for ``norm='per-image'`` z-score.

    Computed on the **pre-conform** input (the caller's original pixels) so pad /
    resize border pixels never skew the stats; the z-score is applied later at the
    norm slot. Pad (reflect/replicate) and bicubic resize are affine-commuting, so
    applying ``(x-m)/s`` with these pre-conform ``(m,s)`` post-conform is identical
    to applying it pre-conform.

    ``x_preconform`` has the batch on dim 0 (any trailing shape). Returns
    ``(mean, std)`` each shape ``(B,)``. ``std`` uses the population (biased)
    estimator to match numpy's ``x.std()`` default and the "subtract mean, divide
    std" intuition. A near-constant sample (``std < eps``) gets ``std = 1`` so the
    divide is a no-op (subtract mean only).
    """
    b = x_preconform.shape[0]
    flat = x_preconform.reshape(b, -1).to(torch.float32)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    std = torch.where(std < _PER_IMAGE_STD_EPS, torch.ones_like(std), std)
    return mean, std


def _warn_values_out_of_unit_range(value: float, *, what: str) -> None:
    """Warn (never raise) if a scalar ``mean`` or ``std`` ∉ ``[0,1]``.

    Norm is scalar-only post-0.1.1, so this takes a single scalar.
    """
    if isinstance(value, (str, bytes, bytearray)):
        return
    try:
        f = float(value)
    except (TypeError, ValueError):
        return
    if f < -_UNIT_RANGE_TOL or f > 1.0 + _UNIT_RANGE_TOL:
        warnings.warn(
            f"{what}={f} is outside [0,1]; omniem normalizes a [0,1]-scaled "
            f"input. If this is a raw 0-255 stat, divide by 255.",
            category=OmniEMWarning,
            stacklevel=3,
        )


def _warn_tensor_out_of_unit_range(x: torch.Tensor, *, what: str) -> None:
    """Warn (never raise) if the scaled input tensor spans outside ``[0,1]``.

    For model / argument norm only — the model's fixed ``mean/std`` assume a
    ``[0,1]`` domain, so an input outside it likely means a wrong ``--scale``.
    Signed-int dtype-default scaling lands in ``[-1,1]`` and trips this by design
    (warn-only).
    """
    if x.numel() == 0:
        return
    mn = float(x.amin())
    mx = float(x.amax())
    if mn < -_UNIT_RANGE_TOL or mx > 1.0 + _UNIT_RANGE_TOL:
        warnings.warn(
            f"{what} spans [{mn:.4g}, {mx:.4g}], outside [0,1]; the model's "
            f"[0,1]-domain normalization may be wrong. Check the input scaling "
            f"(integers should be divided by their dtype max).",
            category=OmniEMWarning,
            stacklevel=3,
        )


# Accepted axes characters. ``b z y x`` is the production set; ``c`` (channel) is
# **rejected** under the gray0 contract — the encoder synthesises the
# channel layout itself (see ``docs/input-format.md``).
_AXES_VALID: frozenset[str] = frozenset({"b", "z", "y", "x"})


# --------------------------------------------------------------------------------------
# Stage 1: prep (axes fold → conform → scalar norm) → channel-less ``[b, z, y, x]``.
# --------------------------------------------------------------------------------------


def prepare_encoder_input(
    backbone: nn.Module,
    x: torch.Tensor,
    *,
    axes: str,
    mean: float,
    std: float,
    norm: None | str | Mapping[str, float] = None,
    conform: str = "strict",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Run the prep stage — return a channel-less canonical ``[b, z, y, x]`` tensor.

    The prep-only half of the encoder forward: axes validate + canonicalise + XY
    conform + scalar mean/std affine. The :class:`~omniem.encoders.base.EMEncoder`
    wrapper calls this from ``_apply_input`` (and from :meth:`EMEncoder.run`).

    Args mirror :func:`compute_encoder` except the return-flag / callback knobs
    (those belong to compute). ``conform`` defaults to ``'strict'`` (matching
    :meth:`EMEncoder.run`); the only other accepted value is ``'resize'``
    (``'pad'`` is rejected).

    Returns:
        ``(canonical, orig_yx)`` — ``canonical`` is the channel-less, normalised
        ``[B, Z, Y, X]`` tensor (Z explicit, never folded here); ``orig_yx`` is the
        caller's pre-conform ``(Y, X)``.
    """
    return _prepare_input(
        x=x,
        axes=axes,
        mean=mean,
        std=std,
        patch_size=int(backbone.patch_size),
        norm=norm,
        conform=conform,
    )


# --------------------------------------------------------------------------------------
# Stage 2: compute (validate → fold + synth → blocks → un-fold → squeeze).
# --------------------------------------------------------------------------------------


def compute_encoder(
    backbone: nn.Module,
    tensor: torch.Tensor,
    *,
    return_cls: bool = True,
    return_patch: bool = False,
    return_blocks: Sequence[int] | None = None,
    block_callback: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
    squeeze: str = "",
) -> dict[str, Any]:
    """Run the compute stage on a canonical channel-less ``[b, z, y, x]`` tensor.

    Strict SHAPE validation (float, rank 4, square + patch-aligned XY) — the
    normalisation is never inferred. Reads ``(B, Z)``, folds ``B*Z`` for the 2D
    blocks, synthesises the channel axis (gray → ``in_chans``), runs the block
    loop (``block_callback`` sees the folded ``[B*Z, tokens, dim]``), then un-folds
    each output to an explicit ``[B, Z, …]`` and applies ``squeeze`` (drop a
    singleton ``b`` / ``z``).

    Output shapes (before ``squeeze``): ``cls [B, Z, D]``, ``patch [B, Z, N, D]``,
    ``inner[i] [B, Z, T, D]``. ``squeeze='bz'`` on a 2D tile (``B=Z=1``) yields
    ``cls [D]`` / ``patch [N, D]``.

    Raises:
        InputContractError: a non-tensor / **integer** tensor (scale is never in
            the package); a non-canonical shape; an unrequested output; a
            non-bool return flag; an out-of-range ``return_blocks`` index; a
            shape-changing ``block_callback``; or an invalid ``squeeze``.
    """
    # return_* bools must be real bools — a footgun like ``return_patch="false"``
    # (truthy string) or ``return_cls=[]`` (falsy) must not silently flip behaviour.
    for _name, _val in (("return_cls", return_cls), ("return_patch", return_patch)):
        if not isinstance(_val, bool):
            raise InputContractError(
                f"`{_name}` must be a bool (got {type(_val).__name__}: {_val!r})"
            )

    chosen_blocks = _validate_blocks(return_blocks, depth=backbone.n_blocks)
    if not (return_cls or return_patch or chosen_blocks):
        raise InputContractError(
            "compute_encoder: no output requested — set at least one of "
            "return_cls / return_patch / return_blocks=[...]."
        )
    drop = parse_squeeze(squeeze)
    use_callback = block_callback is not None

    # --- validate the canonical channel-less [b, z, y, x] tensor ----------------
    if not isinstance(tensor, torch.Tensor):
        raise InputContractError(
            f"EMEncoder.forward expects a torch.Tensor (got {type(tensor).__name__})."
        )
    if not torch.is_floating_point(tensor):
        raise InputContractError(
            f"EMEncoder.forward expects a FLOAT canonical [b, z, y, x] tensor "
            f"(got dtype={tensor.dtype}); the package never scales int→float "
            f"(uint8 != ÷255). Use run(image, axes=...) for raw input, or cast first."
        )
    if tensor.ndim != 4:
        raise InputContractError(
            f"EMEncoder.forward expects a canonical 4D [b, z, y, x] tensor "
            f"(got ndim={tensor.ndim}, shape={tuple(tensor.shape)}). Build the "
            f"canonical layout (z=1 for a 2D tile) or use run(image, axes=...)."
        )
    B, Z, Y, X = (int(d) for d in tensor.shape)
    if B <= 0 or Z <= 0 or Y <= 0 or X <= 0:
        raise InputContractError(
            f"EMEncoder.forward: empty axis (got B={B}, Z={Z}, Y={Y}, X={X})."
        )
    patch_size = int(backbone.patch_size)
    if Y != X or Y % patch_size != 0:
        raise InputContractError(
            f"EMEncoder.forward: XY must be square + a multiple of patch_size="
            f"{patch_size} (got {Y}x{X}); conform via run(image, axes=..., "
            f"conform='resize')."
        )

    # --- fold (B, Z) → B*Z, synthesise the channel axis (gray → in_chans) -------
    in_chans = _resolve_in_chans(backbone)
    x4d = tensor.reshape(B * Z, 1, Y, X).repeat(1, in_chans, 1, 1)

    tokens = backbone.prepare_tokens_with_masks(x4d)
    inner: dict[int, torch.Tensor] = {}
    for i, blk in enumerate(backbone.blocks):
        tokens = blk(tokens)
        # Record the tap BEFORE the callback: ``inner[i]`` is the raw post-block,
        # pre-final-norm feature; a head adapter callback must not contaminate it.
        if i in chosen_blocks:
            inner[i] = tokens.clone()
        if use_callback:
            pre_shape = tokens.shape
            tokens = block_callback(i, tokens)
            if not isinstance(tokens, torch.Tensor):
                raise InputContractError(
                    f"block_callback({i}, ...) returned {type(tokens).__name__}, "
                    f"expected torch.Tensor"
                )
            if tokens.shape != pre_shape:
                raise InputContractError(
                    f"block_callback({i}, ...) is not shape-preserving: "
                    f"got {tuple(tokens.shape)}, expected {tuple(pre_shape)}"
                )

    x_norm = backbone.norm(tokens)
    n_reg = int(backbone.num_register_tokens)

    out: dict[str, Any] = {}
    if return_cls:
        # [B*Z, D] → [B, Z, D].
        out["cls"] = _unfold_bz(x_norm[:, 0], B=B, Z=Z, drop=drop)
    if return_patch:
        # Patch tokens follow cls + any register tokens (n_reg == 0 for emdinov1).
        # [B*Z, N, D] → [B, Z, N, D].
        out["patch"] = _unfold_bz(x_norm[:, 1 + n_reg :], B=B, Z=Z, drop=drop)
    if chosen_blocks:
        # [B*Z, T, D] → [B, Z, T, D] per tapped block.
        out["inner"] = {i: _unfold_bz(t, B=B, Z=Z, drop=drop) for i, t in inner.items()}
    return out


def _unfold_bz(t: torch.Tensor, *, B: int, Z: int, drop: frozenset[str]) -> torch.Tensor:
    """Un-fold a ``[B*Z, …]`` feature to ``[B, Z, …]`` and apply ``squeeze``.

    ``drop`` (from :func:`parse_squeeze`) names which of ``b`` / ``z`` to remove —
    each only when singleton (else raise; a non-singleton drop would lose data).
    """
    t = t.reshape(B, Z, *t.shape[1:])
    # Drop z first (dim 1) then b (dim 0) so the b index stays valid while squeezing.
    if "z" in drop:
        if Z != 1:
            raise InputContractError(
                f"squeeze='z' requires a singleton z (got Z={Z}); a non-singleton "
                f"depth cannot be squeezed."
            )
        t = t.squeeze(1)
    if "b" in drop:
        if B != 1:
            raise InputContractError(
                f"squeeze='b' requires a singleton batch (got B={B}); a non-singleton "
                f"batch cannot be squeezed."
            )
        t = t.squeeze(0)
    return t


# --------------------------------------------------------------------------------------
# Validation helpers.
# --------------------------------------------------------------------------------------


def _validate_blocks(blocks: Sequence[int] | None, *, depth: int) -> tuple[int, ...]:
    """Validate ``return_blocks=`` indices and return a deduped sorted tuple.

    The dict returned to the caller is keyed by index (order-independent),
    so deduping/sorting here only affects internal iteration. Negative indices are
    rejected explicitly — Python-style ``-1`` indexing would silently work but
    obscures the API surface.

    Raises:
        InputContractError: ``blocks`` is not iterable, an index is negative, or an
            index is out of ``[0, depth)``.
    """
    if blocks is None:
        return ()
    try:
        items = list(blocks)
    except TypeError as e:
        raise InputContractError(f"`return_blocks` must be iterable (got {blocks!r})") from e
    seen: set[int] = set()
    out: list[int] = []
    for b in items:
        if not isinstance(b, int) or isinstance(b, bool):
            raise InputContractError(
                f"`return_blocks` entries must be plain ints (got {type(b).__name__}: {b!r})"
            )
        if b < 0:
            raise InputContractError(f"`return_blocks` index {b} is negative; use 0..{depth - 1}")
        if b >= depth:
            raise InputContractError(
                f"`return_blocks` index {b} is out of range for a depth-{depth} encoder "
                f"(allowed: 0..{depth - 1})"
            )
        if b in seen:
            raise InputContractError(f"Duplicate `return_blocks` index {b}")
        seen.add(b)
        out.append(b)
    out.sort()  # deterministic iteration; dict key set is the same either way
    return tuple(out)


# --------------------------------------------------------------------------------------
# Input prep: axes fold + normalisation + square/patch conform (channel-less).
# --------------------------------------------------------------------------------------


def _resolve_in_chans(backbone: nn.Module) -> int:
    """Best-effort ``in_chans`` lookup.

    The vendored class stores ``in_chans`` on ``patch_embed`` (PatchEmbed). Walking the
    attribute keeps the wrapper portable across future arch families.
    """
    pe = getattr(backbone, "patch_embed", None)
    return int(getattr(pe, "in_chans", 3)) if pe is not None else 3


def _parse_axes(axes: str) -> str:
    """Parse + validate an ``axes`` string.

    Rules (gray0 contract):

    * whitespace is ignored;
    * every character must be in ``{b, z, y, x}`` — ``c`` is rejected (the
      encoder enforces the gray0 contract; see ``docs/input-format.md``);
    * no duplicates;
    * if ``b`` is present, it must be the FIRST axis;
    * ``y`` and ``x`` are both required (the encoder needs an XY plane).

    Returns the cleaned-up axes string.
    """
    if not isinstance(axes, str) or not axes:
        raise InputContractError(f"`axes` must be a non-empty string (got {axes!r})")
    cleaned = "".join(axes.split())
    if not cleaned:
        raise InputContractError(f"`axes` must contain at least one axis (got {axes!r})")
    seen: set[str] = set()
    for ax in cleaned:
        if ax == "c":
            # gray0 contract: the encoder synthesises the backbone's
            # channel layout itself — caller-provided channel axes are rejected.
            raise InputContractError(
                f"`axes` must not contain 'c' — the encoder takes grayscale "
                f"input only (gray0); see docs/input-format.md (got axes={axes!r})."
            )
        if ax not in _AXES_VALID:
            raise InputContractError(
                f"Unknown axis {ax!r} in axes={axes!r}; allowed: {sorted(_AXES_VALID)}"
            )
        if ax in seen:
            raise InputContractError(f"Duplicate axis {ax!r} in axes={axes!r}")
        seen.add(ax)
    if "b" in cleaned and cleaned[0] != "b":
        raise InputContractError(f"`b` must be the leading axis when present (got axes={axes!r})")
    if "y" not in cleaned or "x" not in cleaned:
        raise InputContractError(f"`axes` must include both 'y' and 'x' (got axes={axes!r})")
    return cleaned


def _prepare_input(
    *,
    x: torch.Tensor,
    axes: str,
    mean: float,
    std: float,
    patch_size: int,
    norm: None | str | Mapping[str, float],
    conform: str = "strict",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Axes fold + XY conform + scalar normalisation → ``(canonical, orig_yx)``.

    Pipeline (channel-less throughout):

    1. Validate ``axes`` (gray0; ``c`` rejected) and ``x.ndim == len(axes)``.
    2. Reorder dims into the canonical ``b z y x`` order; insert size-1 dims for
       any axis the caller omitted (so the rest of the pipeline is shape-uniform).
    3. **Conform XY.** ``strict``: validate square + ``Y % patch_size == 0``,
       raise on miss. ``resize``: bicubic interpolate to the next square multiple of
       ``patch_size`` (fold (B,Z)→B*Z for XY-only resize, then unfold). ``pad`` is
       **rejected** by the encoder (padding without restore would leave padded-region
       tokens in patch/inner).
    4. Apply the **scalar** ``norm`` directive (None → arch ``mean``/``std``;
       ``'per-image'`` → per-sample z-score; ``'prenormalized'`` → skip;
       ``{'mean','std'}`` → scalar override). Channel synthesis is a compute concern,
       so the affine runs once on the channel-less grid (identical to per-channel
       because every synthesised channel is a copy).

    Returns ``(canonical[B, Z, Y, X], orig_yx)``.
    """
    axes_clean = _parse_axes(axes)
    if x.ndim != len(axes_clean):
        raise InputContractError(
            f"x.ndim ({x.ndim}) does not match axes={axes!r} (length {len(axes_clean)})"
        )

    # Canonicalise to b z y x by permuting then unsqueezing missing axes.
    canonical = "bzyx"
    perm = [axes_clean.index(ax) for ax in canonical if ax in axes_clean]
    x = x.permute(*perm)
    pos = 0
    for ax in canonical:
        if ax not in axes_clean:
            x = x.unsqueeze(pos)
        pos += 1

    # Now x has shape (B, Z, Y, X) — channel-less.
    B, Z, Y, X = x.shape
    orig_yx = (int(Y), int(X))

    # Empty-axis guard — a zero-sized axis passes ``Y % patch_size == 0`` (strict)
    # and reaches F.interpolate (resize) where it surfaces a raw torch error.
    if B <= 0 or Z <= 0 or Y <= 0 or X <= 0:
        raise InputContractError(
            f"EMEncoder.apply_input: empty axis (got B={B}, Z={Z}, Y={Y}, X={X}); "
            f"inputs must have positive size on every axis."
        )

    # Per-image (norm='per-image'): capture per-sample stats on the PRE-CONFORM,
    # channel-less input. Applied at the norm slot below.
    per_image = isinstance(norm, str) and norm == _PER_IMAGE
    pi_mean: torch.Tensor | None = None
    pi_std: torch.Tensor | None = None
    if per_image:
        pi_mean, pi_std = _per_image_stats(x)

    # Scaled-input [0,1] range warn (model / argument norm only — prenormalized and
    # per-image are exempt). Checked on the pre-conform scaled input.
    if not per_image and norm != _PRENORMALIZED:
        _warn_tensor_out_of_unit_range(x, what="EMEncoder input")

    # --- (3) Conform XY --------------------------------------------------------
    if conform == "pad":
        # The encoder does not support pad (would pollute patch/inner with
        # padded-region tokens since there is no output restore step).
        raise InputContractError(
            "EMEncoder does not support conform='pad' — the encoder returns "
            "patch/inner tokens on the (possibly resized) grid with no restore; "
            "padding would leak padded-region tokens. Use conform='resize' or "
            "'strict' here."
        )
    if conform == "strict":
        if Y != X:
            raise InputContractError(
                f"Encoder requires square XY (EM in-plane isotropy); got Y={Y}, X={X}"
            )
        if patch_size <= 0 or Y % patch_size != 0:
            raise InputContractError(
                f"XY side ({Y}) must be a positive multiple of patch_size ({patch_size})"
            )
    elif conform == "resize":
        if patch_size <= 0:
            raise InputContractError(f"Invalid patch_size {patch_size}")
        # Target square side: next multiple of patch_size that fits the larger of
        # (Y, X). Already-conforming input is a no-op (target == Y == X).
        target = _ceil_to_multiple(max(Y, X), patch_size)
        if (Y, X) != (target, target):
            # Bicubic interpolate XY only — fold (B, Z) → B*Z so the 4D interpolate
            # touches only the last two axes.
            xf = x.reshape(B * Z, 1, Y, X).to(dtype=torch.float32)
            xf = F.interpolate(
                xf,
                size=(target, target),
                mode="bicubic",
                align_corners=False,
            )
            x = xf.reshape(B, Z, target, target).to(dtype=x.dtype)
            Y = X = target
    else:
        raise InputContractError(
            f"conform must be one of 'resize', 'strict' (got {conform!r}); the "
            f"encoder rejects 'pad'."
        )

    # --- (4) Scalar normalisation: the ``norm`` directive -----------------------
    if per_image:
        # Per-sample z-score with the PRE-CONFORM stats (broadcast over B). Stats
        # are float32 for stability; cast the RESULT back to x.dtype so a non-fp32
        # encoder (CUDA fp16) matches the fixed-affine path.
        assert pi_mean is not None and pi_std is not None
        m = pi_mean.to(device=x.device).view(B, 1, 1, 1)
        s = pi_std.to(device=x.device).view(B, 1, 1, 1)
        x = ((x - m) / s).to(x.dtype)
    else:
        eff_mean, eff_std = _resolve_norm(norm, mean, std)
        if eff_mean is not None:  # None,None == 'prenormalized' → skip the affine
            if isinstance(norm, Mapping):  # argument-decide → warn on out-of-[0,1]
                _warn_values_out_of_unit_range(eff_mean, what="norm mean")
                _warn_values_out_of_unit_range(eff_std, what="norm std")
            x = _apply_scalar_affine(x, eff_mean, eff_std)

    return x, orig_yx


def _ceil_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` UP to the next non-zero multiple of ``multiple``.

    Helper for the conform pipelines (both encoder and model). Falls back to
    ``multiple`` when ``value`` is zero so an empty axis still picks a sane
    target (never 0).
    """
    if value <= 0:
        return int(multiple)
    return int(((value + multiple - 1) // multiple) * multiple)


def _resolve_norm(
    norm: None | str | Mapping[str, float],
    mean: float,
    std: float,
) -> tuple[float | None, float | None]:
    """Resolve the ``norm=`` directive into an effective **scalar** ``(mean, std)``.

    Returns ``(None, None)`` for ``'prenormalized'`` (caller signalling "skip the
    affine"); otherwise a concrete scalar ``(mean, std)`` pair.

    Accepted shapes:

    * ``None`` → the arch ``mean`` / ``std`` passed in;
    * ``'prenormalized'`` → ``(None, None)`` (skip);
    * ``{'mean': m, 'std': s}`` → ``(m, s)`` — **both** keys required, each a
      finite scalar (per-channel sequences are rejected: EM is grayscale, the
      synthesised channels are identical, so a per-channel override is meaningless).

    Anything else (a callable, an unknown string, a partial / extra-key dict, a
    sequence ``mean`` / ``std``) raises :class:`InputContractError`.
    """
    if norm is None:
        return mean, std
    if isinstance(norm, str):
        if norm == _PRENORMALIZED:
            return None, None
        raise InputContractError(
            f"`norm` string must be {_PRENORMALIZED!r} or {_PER_IMAGE!r} (got {norm!r}); "
            f"for a custom affine pass norm={{'mean': m, 'std': s}} (scalars), or "
            f"normalise the input yourself and pass norm='{_PRENORMALIZED}'."
        )
    if isinstance(norm, Mapping):
        keys = set(norm)
        if keys != {"mean", "std"}:
            raise InputContractError(
                f"`norm` dict must have exactly the keys {{'mean', 'std'}} "
                f"(got {sorted(keys)}); both are required."
            )
        m, s = norm["mean"], norm["std"]
        if m is None or s is None:
            raise InputContractError(
                f"`norm` dict 'mean'/'std' must not be None (got mean={m!r}, std={s!r})."
            )
        return m, s
    raise InputContractError(
        f"`norm` must be None, '{_PRENORMALIZED}', '{_PER_IMAGE}', or a "
        f"{{'mean','std'}} dict (got {type(norm).__name__}: {norm!r}). Callables "
        f"are not accepted — normalise the input yourself and pass "
        f"norm='{_PRENORMALIZED}'."
    )


def _apply_scalar_affine(
    x: torch.Tensor,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Apply scalar ``(x - mean) / std`` on the channel-less ``[B, Z, Y, X]`` grid."""
    m_t = _to_scalar_tensor(mean, ref=x, name="mean")
    s_t = _to_scalar_tensor(std, ref=x, name="std", positive=True)
    return (x - m_t) / s_t


def _to_scalar_tensor(
    value: float,
    *,
    ref: torch.Tensor,
    name: str,
    positive: bool = False,
) -> torch.Tensor:
    """Validate a **scalar** mean-or-std and return it as a 0-d broadcast tensor.

    Rejects sequences (norm is scalar-only post-0.1.1), strings/bytes, bools, and
    non-finite / non-positive (for ``std``) values.
    """
    if isinstance(value, bool):
        raise InputContractError(f"{name} must be numeric (got bool {value!r})")
    if isinstance(value, (int, float)):
        f = float(value)
        if not math.isfinite(f):
            raise InputContractError(f"{name} must be finite (got {value!r})")
        if positive and f <= 0:
            raise InputContractError(f"{name} must be strictly positive (got {value!r})")
        return torch.tensor(f, dtype=ref.dtype, device=ref.device)
    if isinstance(value, (str, bytes, bytearray)):
        raise InputContractError(f"{name} must be a numeric scalar, not a string (got {value!r})")
    if isinstance(value, Iterable):
        raise InputContractError(
            f"{name} must be a SCALAR (got a sequence {value!r}); per-channel "
            f"mean/std is rejected — EM is grayscale, the synthesised channels are "
            f"identical, so a per-channel affine is meaningless."
        )
    raise InputContractError(
        f"{name} must be a numeric scalar (got {type(value).__name__})"
    )


__all__ = ["compute_encoder", "prepare_encoder_input"]

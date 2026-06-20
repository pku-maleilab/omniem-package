"""omniem-driven forward pass for the DINOv2 backbone.

This module reimplements the backbone's ``forward_features`` loop so it can host
the omniem block hook (``block_callback``) and the public ``inner=`` taps. The
underlying vendored class is **not** edited.

What ``forward_features`` does (compared to upstream):

* drives the loop explicitly (rather than ``model.forward_features``) so a
  ``block_callback(i, x) -> x`` runs **after** block ``i`` and **feeds** block ``i+1``
  The callback is **shape-preserving** — a cheap per-block ``.shape``
  compare raises :class:`~omniem.errors.InputContractError` on a mismatch
  with a clear error.
* lets the caller pick **which fields to return** via three independent flags
  (``return_cls`` / ``return_patch`` / ``return_blocks``) — the prior ``want=``
  tuple + ``blocks=`` selector are gone. All-false (nothing requested) is a
  contract error.
* exposes raw **per-block pre-norm** features via ``return_blocks=[i, j, ...]``;
  the returned ``inner`` is always a dict keyed by block index.
* applies the **grayscale-only ("gray0") input-transform contract**: a ``c`` axis
  in ``axes`` is rejected (the encoder synthesises the backbone's channel layout
  itself — gray0 → repeat to ``in_chans``); then ``(x − mean) / std`` driven by
  ``norm=`` (None → config default, ``'prenormalized'`` → skip, ``{'mean','std'}``
  → override); axes folding
  (``b z y x`` → ``b*z`` so the encoder stays strictly 2D); square + patch-aligned
  XY enforcement. See ``docs/input-format.md`` for the
  user-facing rules.

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

from omniem.errors import InputContractError, OmniEMWarning
from omniem.prepared import Prepared

# ``norm=`` accepts one of four shapes (replacing the old
# mean=/std=/transform= trio):
#   * None            → use the arch ``mean`` / ``std`` (passed in);
#   * 'prenormalized' → skip the affine (the caller already did it);
#   * 'per-image'     → per-sample z-score (x - mean(x)) / std(x);
#   * {'mean', 'std'} → override (both keys required).
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


def _warn_values_out_of_unit_range(value: float | Sequence[float], *, what: str) -> None:
    """Warn (never raise) if a scalar / per-channel ``mean`` or ``std`` ∉ ``[0,1]``.

    Accepts scalar OR per-channel sequence and iterates. One warning
    per call is enough.
    """
    seq: Sequence[float]
    if isinstance(value, (str, bytes, bytearray)):
        return
    seq = value if isinstance(value, Sequence) else [value]  # type: ignore[assignment]
    for v in seq:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f < -_UNIT_RANGE_TOL or f > 1.0 + _UNIT_RANGE_TOL:
            warnings.warn(
                f"{what}={f} is outside [0,1]; omniem normalizes a [0,1]-scaled "
                f"input. If this is a raw 0-255 stat, divide by 255.",
                category=OmniEMWarning,
                stacklevel=3,
            )
            return


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

# Acceptable axes precedence: a leading 'b' (when present) must be the first character
# — anything else would make the fold-z semantics ambiguous.


# --------------------------------------------------------------------------------------
# Public forward entry point.
# --------------------------------------------------------------------------------------


def forward_features(
    backbone: nn.Module,
    x: torch.Tensor,
    *,
    axes: str,
    mean: float,
    std: float,
    return_cls: bool = True,
    return_patch: bool = False,
    return_blocks: Sequence[int] | None = None,
    block_callback: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
    norm: None | str | Mapping[str, float | Sequence[float]] = None,
) -> dict[str, Any]:
    """Run the omniem-driven forward pass on a built backbone.

    See module docstring for the high-level contract. The signature uses
    independent ``return_*`` flags — the prior ``want``/``blocks``
    coupling is gone.

    Args:
        backbone: A built :class:`DinoVisionTransformer`-shaped module. Must expose
            ``patch_embed``, ``prepare_tokens_with_masks``, ``blocks``, ``norm``,
            ``num_register_tokens``, ``patch_size``, ``embed_dim``, ``n_blocks``.
        x: The caller's input tensor. Dim is governed by ``axes``. **Grayscale
            only** — ``axes`` must not contain ``c``; see ``docs/input-format.md``.
        axes: A string of one-character axis tags drawn from ``{b, z, y, x}``
            (whitespace ignored). A ``c`` axis is a contract error. Validation
            rules in :func:`_parse_axes`.
        mean/std: The arch's pretraining normalisation (from
            :func:`omniem.encoders.registry.arch_mean_std`). Used when ``norm`` is
            ``None``; a ``norm`` dict overrides them; ``norm='prenormalized'`` skips
            the affine.
        return_cls: When ``True`` (default), include the cls token in the output.
        return_patch: When ``True``, include the patch tokens.
        return_blocks: Optional sequence of block indices to tap. Each must be in
            ``[0, n_blocks)``; negative indices and out-of-range raise. ``None`` /
            empty → no taps. The returned ``inner`` dict is keyed by block index
            Selecting indices both *requests* and *selects* the taps — the
            former ``want=('inner',)`` requirement is gone.
        block_callback: Optional ``fn(i, x) -> x`` run after block ``i`` and feeding
            block ``i+1`` — must be **shape-preserving** (a per-block ``.shape`` check
            enforces this; B13).
        norm: The normalisation directive (replaces the old
            ``mean=``/``std=``/``transform=`` trio):

            * ``None`` (default) → use the arch ``mean`` / ``std``;
            * ``'prenormalized'`` → skip the affine (caller already normalised);
            * ``{'mean': m, 'std': s}`` → override the affine; **both** keys are
              required (a partial dict raises). ``m`` / ``s`` may be scalars or
              per-channel sequences of length ``in_chans``.

            Anything else (a callable, an unknown string, a partial dict) raises
            :class:`InputContractError`.

    Returns:
        A flat dict. ``cls`` ∈ ``[B, C]`` (when ``return_cls``);
        ``patch`` ∈ ``[B, N_p, C]`` (when ``return_patch``); ``inner`` ∈
        ``{i: Tensor of shape [B*Z, N_total, C]}`` when ``return_blocks`` is
        non-empty. ``B`` here is ``B*Z`` after axes fold — the wrapper
        :class:`EMEncoder` documents this for the caller.

    Raises:
        InputContractError: when ``axes`` contains ``c`` (gray0 contract); when all
            three ``return_*`` flags are falsy (no field requested); when a
            ``return_*`` flag is not a real bool; when ``norm`` is malformed; plus
            the usual input-contract violations (bad axes, bad blocks,
            shape-changing callback).
    """
    # Finding #3: the return_* flags are documented as **bools** — reject non-bool
    # so a footgun like ``return_patch="false"`` (truthy string) or ``return_cls=[]``
    # (falsy) can't silently flip behaviour.
    for _name, _val in (
        ("return_cls", return_cls),
        ("return_patch", return_patch),
    ):
        if not isinstance(_val, bool):
            raise InputContractError(
                f"`{_name}` must be a bool (got {type(_val).__name__}: {_val!r})"
            )

    chosen_blocks = _validate_blocks(return_blocks, depth=backbone.n_blocks)
    # All-false guard (round-2 / B16): silently returning {} would be a bug magnet.
    # At least one of the four return_* flags must be truthy.
    if not (return_cls or return_patch or chosen_blocks):
        raise InputContractError(
            "forward_features: no output requested — set at least one of "
            "return_cls / return_patch / return_blocks=[...]."
        )
    use_callback = block_callback is not None

    # Step 1: axes fold + channel handling + normalisation -----------------------
    prepared = _prepare_input(
        x=x,
        axes=axes,
        mean=mean,
        std=std,
        patch_size=int(backbone.patch_size),
        in_chans=_resolve_in_chans(backbone),
        norm=norm,
    )
    x4d = prepared.tensor

    # Step 2: tokens → blocks → norm (the manual forward_features loop) -----------
    tokens = backbone.prepare_tokens_with_masks(x4d)
    inner: dict[int, torch.Tensor] = {}
    for i, blk in enumerate(backbone.blocks):
        tokens = blk(tokens)
        # Record the tap BEFORE the callback (tap semantics):
        # ``inner[i]`` is the **raw** post-block, pre-final-norm feature. A head
        # adapter callback must NOT contaminate it — otherwise the downstream head
        # sees the same data the adapter already transformed and the contract folds.
        # ``.clone()`` guarantees the tap is independent of every later op.
        if i in chosen_blocks:
            inner[i] = tokens.clone()
        if use_callback:
            # Shape-preserving guard. We capture the pre-callback shape and
            # compare against the post-callback one — anything else trips the
            # InputContractError so downstream blocks never see a malformed grid.
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
        out["cls"] = x_norm[:, 0]
    if return_patch:
        # Patch tokens follow cls + any register tokens. ``n_reg`` is 0 for emdinov1
        # (no registers); the offset keeps the slice correct for the backbone's layout.
        out["patch"] = x_norm[:, 1 + n_reg :]
    if chosen_blocks:
        out["inner"] = inner
    return out


# --------------------------------------------------------------------------------------
# Split surfaces: apply_input + compute on the dinov2 backbone.
# --------------------------------------------------------------------------------------


def apply_input_features(
    backbone: nn.Module,
    x: torch.Tensor,
    *,
    axes: str,
    mean: float,
    std: float,
    norm: None | str | Mapping[str, float | Sequence[float]] = None,
    conform: str = "resize",
) -> Prepared:
    """Run only the input-transform stage; return the :class:`Prepared` carrier.

    This is the prep-only half of :func:`forward_features` — axes fold + gray0
    synthesis + mean/std affine + XY conform. The :class:`EMEncoder` wrapper calls
    this in :meth:`EMEncoder.apply_input`.

    Args mirror :func:`forward_features` except the return-flag / callback knobs
    (those belong to compute, not prep). ``conform`` defaults to ``'resize'`` for
    the encoder; the only other accepted value is ``'strict'`` (``'pad'`` is
    rejected — see :func:`_prepare_input`).
    """
    return _prepare_input(
        x=x,
        axes=axes,
        mean=mean,
        std=std,
        patch_size=int(backbone.patch_size),
        in_chans=_resolve_in_chans(backbone),
        norm=norm,
        conform=conform,
    )


def compute_features(
    backbone: nn.Module,
    prepared: Prepared,
    *,
    return_cls: bool = True,
    return_patch: bool = False,
    return_blocks: Sequence[int] | None = None,
    block_callback: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Run only the compute stage on an already-:class:`Prepared` tensor.

    This is the post-prep half of :func:`forward_features`. The
    :class:`EMEncoder` wrapper calls this when
    :meth:`EMEncoder.forward` is invoked with a :class:`Prepared` as ``x``
    (the type is the signal; no ``prepared=`` bool).

    Args mirror :func:`forward_features`'s return-flag block; ``norm``/``axes``
    are intentionally absent — they were already applied / baked into the
    :class:`Prepared` upstream.
    """
    # Same return-flag bool guard as forward_features.
    for _name, _val in (
        ("return_cls", return_cls),
        ("return_patch", return_patch),
    ):
        if not isinstance(_val, bool):
            raise InputContractError(
                f"`{_name}` must be a bool (got {type(_val).__name__}: {_val!r})"
            )

    chosen_blocks = _validate_blocks(return_blocks, depth=backbone.n_blocks)
    if not (return_cls or return_patch or chosen_blocks):
        raise InputContractError(
            "compute_features: no output requested — set at least one of "
            "return_cls / return_patch / return_blocks=[...]."
        )
    use_callback = block_callback is not None

    # Prepared meta validation — guard against a
    # user-constructed Prepared whose ``conform``/``axes``/``B``/``Z``/``stride``
    # disagree with the tensor it carries. (Pad is unsupported by the encoder,
    # ``stride`` is the cross-layer discriminator, etc.)
    if not isinstance(prepared.axes, str) or not prepared.axes:
        raise InputContractError(
            f"Prepared.axes must be a non-empty string (got {prepared.axes!r})"
        )
    _parse_axes(prepared.axes)
    if prepared.conform not in ("resize", "strict"):
        # The encoder rejects ``pad`` outright; anything else is an
        # unknown mode.
        raise InputContractError(
            f"Encoder Prepared.conform must be 'resize' or 'strict' "
            f"(got {prepared.conform!r})."
        )
    patch_size = int(backbone.patch_size)
    if prepared.stride != patch_size:
        raise InputContractError(
            f"Prepared.stride={prepared.stride} disagrees with the backbone's "
            f"patch_size={patch_size}. Did you feed a model-Prepared (stride=112) "
            f"to the encoder?"
        )

    # Prepared tensor validation. The encoder-prepared
    # carrier MUST be 4D float ``[B*Z, in_chans, S, S]`` with square stride-aligned XY.
    x4d = prepared.tensor
    if not isinstance(x4d, torch.Tensor):
        raise InputContractError(
            f"Prepared.tensor must be a torch.Tensor (got {type(x4d).__name__})"
        )
    if not torch.is_floating_point(x4d):
        raise InputContractError(
            f"Prepared.tensor must be floating-point (got dtype={x4d.dtype})"
        )
    if x4d.ndim != 4:
        raise InputContractError(
            f"Encoder-prepared tensor must be 4D [B*Z, C, S, S] (got ndim={x4d.ndim}, "
            f"shape={tuple(x4d.shape)})"
        )
    BZ, C_in, Sy, Sx = x4d.shape
    expected_c = _resolve_in_chans(backbone)
    if C_in != expected_c:
        raise InputContractError(
            f"Prepared.tensor channel count {C_in} does not match in_chans={expected_c}"
        )
    if Sy != Sx or Sy <= 0 or Sy % patch_size != 0:
        raise InputContractError(
            f"Prepared.tensor XY must be square + multiple of patch_size={patch_size} "
            f"(got {Sy}x{Sx})"
        )
    # B * Z must equal the leading batch dim — the encoder folds (B, Z) → B*Z
    # in apply_input, so this disagrees only when the meta has been corrupted.
    if int(prepared.B) * int(prepared.Z) != BZ:
        raise InputContractError(
            f"Prepared.B * Prepared.Z ({prepared.B} * {prepared.Z} = "
            f"{prepared.B * prepared.Z}) does not match tensor leading dim {BZ}."
        )

    tokens = backbone.prepare_tokens_with_masks(x4d)
    inner: dict[int, torch.Tensor] = {}
    for i, blk in enumerate(backbone.blocks):
        tokens = blk(tokens)
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
        out["cls"] = x_norm[:, 0]
    if return_patch:
        # Patch tokens follow cls + any register tokens (n_reg == 0 for emdinov1).
        out["patch"] = x_norm[:, 1 + n_reg :]
    if chosen_blocks:
        out["inner"] = inner
    return out


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
# Input prep: axes fold + channel handling + normalisation + square/patch check.
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
                f"`axes` must not contain 'c' — EMEncoder.forward takes grayscale "
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
    in_chans: int,
    norm: None | str | Mapping[str, float | Sequence[float]],
    conform: str = "strict",
) -> Prepared:
    """Apply axes fold + channel synthesis + normalisation + XY conform → ``Prepared``.

    Pipeline:

    1. Validate ``axes`` (gray0; ``c`` rejected) and verify ``x.ndim == len(axes)``.
    2. Reorder dims into the canonical ``b z y x`` order; insert size-1 dims for
       any axis the caller omitted (so the rest of the pipeline is shape-uniform).
    3. **Conform XY.** ``strict``: validate square + ``Y % patch_size == 0``,
       raise on miss. ``resize``: bicubic interpolate to the next square multiple of
       ``patch_size`` (fold (B,Z)→B*Z for XY-only resize, then unfold). ``pad`` is
       **rejected** by the encoder (padding without restore would leave padded-region
       tokens in patch/inner).
    4. **Synthesise the channel axis**: unsqueeze a length-1 ``C`` dim and
       ``repeat`` it to ``in_chans``. The encoder *always* sees
       ``[B, Z, in_chans, Y, X]`` (EM is grayscale — there is no channel knob).
    5. Apply the ``norm`` directive (None → arch ``mean``/``std``;
       ``'prenormalized'`` → skip; ``{'mean','std'}`` → override).
    6. Fold (b, z) → leading ``b*z`` batch dim.

    Returns a frozen :class:`Prepared` carrying the canonical 4D tensor plus
    the metadata downstream consumers (compute, optional un-conform) need.
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

    # Empty-axis guard — a zero-sized axis passes
    # ``Y % patch_size == 0`` (strict) and reaches F.pad/interpolate (resize)
    # where it surfaces a raw torch error. Reject at the wrapper with a clear
    # message.
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
    pad_or_scale: dict = {}
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
        pad_or_scale = {"target": int(target)}
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
            f"conform must be one of 'pad', 'resize', 'strict' (got {conform!r})"
        )

    # --- (4) Channel synthesis (gray0 → in_chans) -------------------------------
    x = x.unsqueeze(2).repeat(1, 1, in_chans, 1, 1)
    C_in = in_chans

    # --- (5) Normalisation: the ``norm`` directive ------------------------------
    if per_image:
        # Per-sample z-score with the PRE-CONFORM stats (broadcast over B). Stats are
        # float32 for stability; cast the RESULT back to x.dtype so a non-fp32 encoder
        # (e.g. CUDA fp16) matches the fixed-affine path and the raw forward (which feeds
        # this Prepared straight into the backbone with no re-cast) does not diverge from
        # the split path.
        assert pi_mean is not None and pi_std is not None
        m = pi_mean.to(device=x.device).view(B, 1, 1, 1, 1)
        s = pi_std.to(device=x.device).view(B, 1, 1, 1, 1)
        x = ((x - m) / s).to(x.dtype)
    else:
        eff_mean, eff_std = _resolve_norm(norm, mean, std)
        if eff_mean is not None:  # None,None == 'prenormalized' → skip the affine
            if isinstance(norm, Mapping):  # argument-decide → warn on out-of-[0,1]
                _warn_values_out_of_unit_range(eff_mean, what="norm mean")
                _warn_values_out_of_unit_range(eff_std, what="norm std")
            x = _apply_affine(x, eff_mean, eff_std, c=C_in)

    # --- (6) Fold (B, Z) → leading B*Z; drop the now-singleton Z dim ------------
    # Shape goes from (B, Z, C, Y, X) → (B*Z, C, Y, X).
    x = x.reshape(B * Z, C_in, Y, X)
    return Prepared(
        tensor=x,
        axes=axes_clean,
        conform=conform,  # type: ignore[arg-type]
        orig_yx=orig_yx,
        pad_or_scale=pad_or_scale,
        B=int(B),
        Z=int(Z),
        stride=int(patch_size),
    )


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
    norm: None | str | Mapping[str, float | Sequence[float]],
    mean: float,
    std: float,
) -> tuple[float | Sequence[float] | None, float | Sequence[float] | None]:
    """Resolve the ``norm=`` directive into effective ``(mean, std)``.

    Returns ``(None, None)`` for ``'prenormalized'`` (caller signalling "skip the
    affine"); otherwise a concrete ``(mean, std)`` pair to feed
    :func:`_apply_affine`.

    Accepted shapes:

    * ``None`` → the arch ``mean`` / ``std`` passed in;
    * ``'prenormalized'`` → ``(None, None)`` (skip);
    * ``{'mean': m, 'std': s}`` → ``(m, s)`` — **both** keys required, neither
      may itself be ``None``.

    Anything else (a callable, an unknown string, a partial / extra-key dict)
    raises :class:`InputContractError`.
    """
    if norm is None:
        return mean, std
    if isinstance(norm, str):
        if norm == _PRENORMALIZED:
            return None, None
        raise InputContractError(
            f"`norm` string must be {_PRENORMALIZED!r} (got {norm!r}); for a custom "
            f"affine pass norm={{'mean': m, 'std': s}}, or normalise the input "
            f"yourself and pass norm='{_PRENORMALIZED}'."
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
        f"`norm` must be None, '{_PRENORMALIZED}', or a {{'mean','std'}} dict "
        f"(got {type(norm).__name__}: {norm!r}). Callables are not accepted — "
        f"normalise the input yourself and pass norm='{_PRENORMALIZED}'."
    )


def _apply_affine(
    x: torch.Tensor,
    mean: float | Sequence[float],
    std: float | Sequence[float],
    *,
    c: int,
) -> torch.Tensor:
    """Apply ``(x - mean) / std`` along the channel axis (dim=2 in our canonical form).

    ``mean``/``std`` may be scalars or per-channel sequences of length ``c``. The
    helper validates the per-channel length so a mismatched override surfaces here
    rather than as a downstream broadcasting nightmare.
    """
    m_t = _to_param_tensor(mean, c=c, ref=x, name="mean")
    s_t = _to_param_tensor(std, c=c, ref=x, name="std", positive=True)
    # x shape is (B, Z, C, Y, X); broadcast mean/std as (1, 1, C, 1, 1).
    return (x - m_t) / s_t


def _to_param_tensor(
    value: float | Sequence[float],
    *,
    c: int,
    ref: torch.Tensor,
    name: str,
    positive: bool = False,
) -> torch.Tensor:
    """Reshape + validate a scalar / per-channel mean-or-std into a broadcast tensor.

    Validates finiteness (and strict positivity when ``positive`` — std must be > 0
    so the affine never divides by zero / produces inf/nan). Rejects strings/bytes
    and non-numeric sequences with a clean :class:`InputContractError` (the old
    ``InputTransform`` Pydantic model used to guard this).
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        if not math.isfinite(f):
            raise InputContractError(f"{name} must be finite (got {value!r})")
        if positive and f <= 0:
            raise InputContractError(f"{name} must be strictly positive (got {value!r})")
        return torch.tensor(f, dtype=ref.dtype, device=ref.device)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        try:
            vec = [float(x) for x in value]
        except (TypeError, ValueError) as e:
            raise InputContractError(
                f"{name} per-channel sequence must be numeric (got {value!r})"
            ) from e
        if len(vec) != c:
            raise InputContractError(
                f"{name} must be a scalar or length-{c} per-channel sequence (got len {len(vec)})"
            )
        if not all(math.isfinite(x) for x in vec):
            raise InputContractError(f"{name} values must be finite (got {value!r})")
        if positive and any(x <= 0 for x in vec):
            raise InputContractError(f"{name} values must be strictly positive (got {value!r})")
        # Shape (1, 1, C, 1, 1) so it broadcasts against (B, Z, C, Y, X).
        return torch.tensor(vec, dtype=ref.dtype, device=ref.device).reshape(1, 1, c, 1, 1)
    raise InputContractError(
        f"{name} must be a scalar or numeric per-channel sequence (got {type(value).__name__})"
    )


__all__ = ["apply_input_features", "compute_features", "forward_features"]

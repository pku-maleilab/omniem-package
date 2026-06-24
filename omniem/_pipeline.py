"""Shared input/output-pipeline primitives (channel-less ``run`` + canonical compute).

The 0.1.1 *unify-io* redesign removed the ``Prepared`` carrier: ``_apply_input``
now returns a **plain canonical tensor** ``[b, z, y, x]`` plus the pre-conform
``orig_yx``; ``predict`` / ``forward`` consume that canonical tensor directly; and
``run`` threads the recovery args internally. This module holds the few cross-layer
constants/helpers the encoder (:mod:`omniem.encoders`) and the model
(:mod:`omniem.models`) still share, so neither layer has to import the other:

* :data:`ConformMode` — the conform vocabulary. The model accepts all three; the
  encoder only ``'resize'`` / ``'strict'`` (``'pad'`` would leave padded-region
  tokens in patch/inner with no restore).
* :func:`channel_insert` — where the predicted output channel ``c_out`` sits in a
  caller-axes logits tensor: right after ``b`` if ``axes`` has a ``b``, else axis
  0. This is the single source of truth shared by ``_restore`` and the model's
  output-layout code (it replaces the retired ``channel_axis_from_axes``).
* :func:`parse_squeeze` — validate the ``squeeze`` directive (a subset of
  ``{b, z}``; ``c`` / ``x`` / ``y`` and duplicates are rejected). The
  "axis must exist and be singleton" check is applied per-layer at squeeze time,
  where the concrete sizes are known.
"""

from __future__ import annotations

from typing import Literal

from omniem.errors import InputContractError

# Conform vocabulary. The model accepts all three; the encoder accepts only
# ``'resize'`` / ``'strict'``.
ConformMode = Literal["pad", "resize", "strict"]


def channel_insert(axes: str) -> int:
    """Index where the predicted ``c_out`` channel is inserted in caller-axes logits.

    ``axes`` is channel-less (the public input contract). The predicted output
    channel always sits **right after a leading ``b``** if present, else at the
    front — independent of the spatial order: ``'yx' → 0`` (``[C, Y, X]``),
    ``'zyx' → 0`` (``[C, Z, Y, X]``), ``'byx' → 1`` (``[B, C, Y, X]``),
    ``'bzyx' → 1`` (``[B, C, Z, Y, X]``).

    Args:
        axes: The (channel-less) axes string the caller passed. Lowercase.

    Returns:
        ``1`` if ``axes`` starts with ``'b'``, else ``0``.
    """
    if not isinstance(axes, str):
        raise TypeError(f"axes must be a str (got {type(axes).__name__})")
    return 1 if axes.startswith("b") else 0


def parse_squeeze(squeeze: str | None) -> frozenset[str]:
    """Validate the ``squeeze`` directive and return the set of axes to drop.

    ``squeeze`` is a subset of ``{b, z}`` (whitespace ignored). Each listed axis
    is dropped from the restored / feature layout **iff** it exists there and is
    singleton — that existence/singleton check is the caller's (it needs the
    concrete sizes). Here we only reject structurally-invalid directives: a ``c``
    / ``x`` / ``y`` request (those axes are never squeezable), an unknown
    character, or a duplicate.

    Args:
        squeeze: The directive (``""`` / ``None`` → drop nothing).

    Returns:
        A frozenset drawn from ``{'b', 'z'}``.

    Raises:
        InputContractError: ``squeeze`` is not a string, names ``c`` / ``x`` /
            ``y``, contains an unknown axis, or repeats an axis.
    """
    if squeeze is None:
        return frozenset()
    if not isinstance(squeeze, str):
        raise InputContractError(
            f"`squeeze` must be a string subset of {{'b','z'}} "
            f"(got {type(squeeze).__name__}: {squeeze!r})"
        )
    cleaned = "".join(squeeze.split())
    seen: set[str] = set()
    for ch in cleaned:
        if ch in ("c", "x", "y"):
            raise InputContractError(
                f"`squeeze` may only drop the batch/depth axes 'b'/'z' — {ch!r} is "
                f"not squeezable (got squeeze={squeeze!r})."
            )
        if ch not in ("b", "z"):
            raise InputContractError(
                f"Unknown `squeeze` axis {ch!r}; allowed: 'b', 'z' (got squeeze={squeeze!r})."
            )
        if ch in seen:
            raise InputContractError(f"Duplicate axis {ch!r} in squeeze={squeeze!r}.")
        seen.add(ch)
    return frozenset(seen)


__all__ = ["ConformMode", "channel_insert", "parse_squeeze"]

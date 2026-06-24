"""Owner-frozen model-arch registry.

Mirrors :mod:`omniem.encoders.registry`: a public ``arch`` name (e.g.
``"omniemv1"``) selects a one-arg factory that wraps :class:`OmniEMV1Net` with
the pinned constants baked in. End users **cannot change** the per-arch
constants (``feature_size=16``, ``upsample_method="resize"``, STAdapter
``channels=128``, etc.) — adding a new model architecture is a one-line edit
by the omniem maintainer to register a new entry.

A model is fully specified by **(config, weights)** — there is no model bundle
or meta block. The config carries the arch key, the encoder arch name, the
training mean/std, and the per-head shape (``img_z``/``kernel3d_z``/
``out_channels``/``resize4emdino``); the registry factory
turns that recipe into an :class:`OmniEMV1Net`.

Tests register a private arch (e.g. ``"_test_tiny"``) via the same
:func:`register_arch` helper used by the encoder registry; ``list_models``
filters underscore-prefixed names out of the user-facing catalog.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omniem.errors import ConfigError
from omniem.models.omniemv1_net import OmniEMV1Net

# A model-arch factory takes the omniem-side :class:`omniem.config.ModelConfig`
# and returns a built :class:`OmniEMV1Net`. It also accepts an optional ``encoder=``
# keyword: when given, the factory injects that pre-built backbone instead of
# building one (the shared-encoder "borrow" path); when omitted it builds the
# encoder from ``config.encoder``. The factory owns the pinned constants for that
# arch; the config carries the per-head shape (img_z / kernel3d_z / out_channels).
ModelArchFactory = Callable[..., OmniEMV1Net]


@dataclass(frozen=True)
class ModelArchInfo:
    """One model-arch entry: the factory + catalog text shown by ``omniem list-models``.

    Attributes:
        factory: One-arg callable taking a :class:`~omniem.config.ModelConfig`
            and returning a built :class:`OmniEMV1Net`. The factory owns the
            pinned constants the user does NOT configure.
        description: Short single-line description for the catalog.
        stride: The arch's input divisor (the smallest XY side
            that is square + valid for both the encoder ViT patch and the
            STAdapter ``omniem_patch``; for ``omniemv1`` =
            ``lcm(14, 16) = 112``). Discoverable via
            ``model_arch_info(name).stride``; the model's :meth:`apply_input`
            conform/un-conform reads it. ``1`` (identity) is the default for
            test arches.
    """

    factory: ModelArchFactory
    description: str = ""
    stride: int = 1


# --------------------------------------------------------------------------------------
# Built-in factories.
# --------------------------------------------------------------------------------------


def _build_omniemv1(config, *, encoder=None) -> OmniEMV1Net:  # noqa: ANN001 — Pydantic ModelConfig
    """Build the ``omniemv1`` OmniEMV1Net.

    The factory wires the UNETR head per the config's per-head shape: ``img_z`` /
    ``kernel3d_z`` (decides 2D vs 3D + z-kernel), ``out_channels`` (decoder out),
    ``resize4emdino`` (the resize-to-encoder-grid flag). The model returns pure
    logits — activation is owned by :meth:`OmniEM.run` (output stage), not the net.

    The backbone is either **injected** via ``encoder`` (the shared-encoder borrow
    path — the caller already built it, e.g. ``EMEncoder.vit``) or built fresh from
    ``config.encoder`` (the ARCH_REGISTRY name) when ``encoder is None``.

    Constants the user does NOT touch (baked into the arch):
    ``feature_size=16``, ``upsample_method="resize"``, STAdapter
    ``channels=128``, ``dropout=0.1``, ``norm_name="instance"``,
    ``conv_block=True``, ``res_block=False``, ``kernel_xy=3``.
    """
    from omniem.encoders.dinov2.build import build  # local: encoders is a sibling

    backbone = encoder if encoder is not None else build(config.encoder)
    return OmniEMV1Net(
        backbone,
        out_channels=config.out_channels,
        img_z=config.img_z,
        kernel3d_z=config.kernel3d_z,
        resize4emdino=config.resize4emdino,
    )


# --------------------------------------------------------------------------------------
# The owner-frozen model-arch registry.
# --------------------------------------------------------------------------------------


# Owner-frozen omniemv1 input divisor. Mirrors ``omniem.models.XY_SIDE_DIVISOR``
# so the registry has a single source (an anti-drift test asserts they match).
_OMNIEMV1_STRIDE: int = 112

MODEL_ARCH_REGISTRY: dict[str, ModelArchInfo] = {
    "omniemv1": ModelArchInfo(
        factory=_build_omniemv1,
        stride=_OMNIEMV1_STRIDE,
        description=(
            "OmniEM v1 — UNETR-style head over an EM-DINO ViT "
            "backbone, with STAdapter z-fusion."
        ),
    ),
}


# --------------------------------------------------------------------------------------
# Public catalog helpers (mirror omniem.encoders.registry).
# --------------------------------------------------------------------------------------


def list_models() -> list[str]:
    """Sorted list of public model-arch names.

    "Private" archs whose name begins with ``_`` (e.g. the test-only
    ``_test_tiny_model``) are filtered out — they remain buildable through
    :func:`model_arch_info` but never leak into the user-facing catalog.
    """
    return sorted(name for name in MODEL_ARCH_REGISTRY if not name.startswith("_"))


def model_arch_info(name: str) -> ModelArchInfo:
    """Look up the :class:`ModelArchInfo` for ``name``.

    Raises:
        ConfigError: When ``name`` is not registered. The available archs are
            listed in the message so the caller can fix a typo.
    """
    info = MODEL_ARCH_REGISTRY.get(name)
    if info is None:
        raise ConfigError(
            f"Unknown model arch {name!r}; registered archs: {list_models()}"
        )
    return info


# --------------------------------------------------------------------------------------
# Internal test hooks (NOT public API).
# --------------------------------------------------------------------------------------


def register_arch(
    name: str,
    factory: ModelArchFactory,
    *,
    description: str = "",
    stride: int = 1,
) -> None:
    """Register a new model arch factory. Test-only — duplicates raise."""
    if name in MODEL_ARCH_REGISTRY:
        raise ConfigError(f"model arch {name!r} is already registered")
    MODEL_ARCH_REGISTRY[name] = ModelArchInfo(
        factory=factory,
        description=description,
        stride=stride,
    )


def unregister_arch(name: str) -> None:
    """Remove a model-arch entry (test-only). Missing entry is a no-op."""
    MODEL_ARCH_REGISTRY.pop(name, None)


__all__ = [
    "MODEL_ARCH_REGISTRY",
    "ModelArchFactory",
    "ModelArchInfo",
    "list_models",
    "model_arch_info",
    "register_arch",
    "unregister_arch",
]

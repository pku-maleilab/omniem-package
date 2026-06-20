"""Owner-frozen architecture registry for omniem encoders.

The arch registry maps a public ``arch`` name (string) to an :class:`ArchInfo`
record carrying its zero-arg backbone factory and a small catalog tuple
(``description``, ``weights_url``) for ``omniem list-encoders``. End users
**cannot change** the hyperparameters baked into a factory — adding a new
pretrained encoder is a one-line edit by the omniem maintainer to register a
new entry. An encoder is selected by ``arch`` name and loaded with a weights
path (``EMEncoder.load(arch, weights)``) — there is no encoder config object.
Each entry also carries the arch's pretraining ``mean``/``std`` (the encoder's
frozen normalisation).

This is the "encoder-agnostic by brand" pillar of the design: a future DINOv3
swap will land as a new factory + ARCH_REGISTRY entry (and possibly a new
``encoders/<family>/backbone.py``); the public API surface —
:class:`omniem.encoders.EMEncoder` and ``enc.forward(...)`` — stays unchanged.

Design note — ``ArchInfo`` vs the previous ``ArchFactory``
----------------------------------------------------------

The registry maps an arch name to a zero-arg ``ArchFactory = Callable[[],
nn.Module]`` plus a small struct **for user-facing catalog text** (description +
weights_url shown by
``omniem list-encoders``), not build args: the catalog is a static surface that
users read; the factory is still the only behaviour.

Test extension hook
-------------------

Tests register a ``_test_tiny`` arch via a session fixture in
``tests/conftest.py``. The internal helpers :func:`register_arch` and
:func:`unregister_arch` exist for that purpose and are NOT part of the public
package API; the package itself never re-registers entries at runtime. Outside
of tests, ``ARCH_REGISTRY`` is treated as a read-only constant.

The ``_test_tiny`` arch starts with an underscore; :func:`list_encoders`
filters such "private" arch names out, so the test arch never leaks into the
user-facing catalog while remaining buildable through :func:`arch_info` /
:func:`omniem.encoders.dinov2.build.build`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from omniem.encoders.dinov2.backbone import DinoVisionTransformer
from omniem.errors import ConfigError

# Type alias for an arch factory: zero-arg callable returning a built backbone.
ArchFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class ArchInfo:
    """One arch entry: the factory + catalog text shown by ``omniem list-encoders``.

    Attributes:
        factory: Zero-arg callable returning a freshly built ``nn.Module``.
        mean: The arch's **pretraining** normalisation mean. This is a fixed fact
            of the pretrained encoder (like ``patch_size`` / ``embed_dim``), NOT a
            user knob — the encoder was trained on this normalisation, so it is
            owner-frozen here rather than hardcoded as a package-global default.
            ``EMEncoder.load(arch, …)`` reads it as the encoder's effective
            normalisation. Default ``0.0`` (identity) for an arch that declares no
            normalisation.
        std: The arch's pretraining normalisation std (see ``mean``). Default
            ``1.0`` (identity).
        description: Short single-line description for the catalog
            (e.g. ``"EM-DINOv2 ViT-L/14 — EM-domain pretrained encoder
            (pretrain dataset: v3)."``).
        weights_url: Public URL to fetch the corresponding raw ``vit.*``
            checkpoint; an empty string is rendered as ``"not yet available"``
            by the CLI.
    """

    factory: ArchFactory
    # The attribute name the EMEncoder wrapper holds the backbone under — which
    # drives the state_dict key prefix. ``"vit"`` for ViT archs (so a raw
    # ``vit.*`` checkpoint loads via strict=True with no key rename); a future
    # non-ViT family declares its own (e.g. ``"mamba"``). The general EMEncoder
    # reads this from the arch instead of hard-coding ``vit``.
    backbone_attr: str = "vit"
    mean: float = 0.0
    std: float = 1.0
    # The arch's input-divisor (ViT patch_size). Discoverable via
    # ``arch_info(name).stride``; the encoder's apply_input enforces
    # ``Y == X`` and ``Y % stride == 0`` and the resize-conform path interpolates
    # to a square multiple of this value. ``1`` is the identity / no-divisor
    # default used by test arches (degenerate but harmless).
    stride: int = 1
    description: str = ""
    weights_url: str = ""


# --------------------------------------------------------------------------------------
# Built-in factories.
# --------------------------------------------------------------------------------------


def _build_emdinov1() -> nn.Module:
    """Build the EM-DINOv2-finetuned ViT-L (emdinov1) backbone.

    Hyperparameters are the ViT-L/14 pretraining config plus the training
    presets carried into the saved checkpoint (``drop_path_rate=0.3``,
    ``drop_path_uniform=True``). Domain names map to constructor-native ones:
    ``layerscale`` → ``init_values``, ``pose_size`` → ``img_size``.

    ``arch`` identifies the architecture; the pretraining-dataset identifier
    (``v3``) is documentation only, not encoded in the weight file (no meta
    block).
    """
    return DinoVisionTransformer(
        img_size=518,  # = pose_size in the source's domain naming
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop_path_rate=0.3,
        drop_path_uniform=True,
        init_values=1.0e-5,  # = layerscale
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )


# --------------------------------------------------------------------------------------
# The owner-frozen registry. Editing this dict outside of tests requires an omniem
# release.
# --------------------------------------------------------------------------------------


ARCH_REGISTRY: dict[str, ArchInfo] = {
    "emdinov1": ArchInfo(
        factory=_build_emdinov1,
        # EM-DINO pretraining normalisation (image space [0, 1]). Owner-frozen —
        # this is the stat the encoder was trained with, not a user default.
        mean=0.595446,
        std=0.211906,
        # The ViT patch size — the encoder's input divisor. Anti-drift
        # tests pin this against the built backbone's ``patch_size``.
        stride=14,
        description=("EM-DINOv2 ViT-L/14 — EM-domain pretrained encoder (pretrain dataset: v3)."),
        # An empty string renders as "not yet available"; this literal
        # "waiting soon …" is shown verbatim by the catalog instead. Stored
        # here so the catalog output stays deterministic.
        weights_url="waiting soon …",
    ),
}


# --------------------------------------------------------------------------------------
# Public catalog helpers.
# --------------------------------------------------------------------------------------


def list_encoders() -> list[str]:
    """Sorted list of public arch names.

    "Private" archs whose name begins with ``_`` (e.g. the test-only
    ``_test_tiny``) are filtered out — they remain buildable through
    :func:`arch_info` / :func:`omniem.encoders.dinov2.build.build`, but never
    leak into the user-facing catalog.
    """
    return sorted(name for name in ARCH_REGISTRY if not name.startswith("_"))


def arch_info(name: str) -> ArchInfo:
    """Look up the :class:`ArchInfo` for ``name``.

    Raises:
        ConfigError: When ``name`` is not registered. The available archs are
            listed in the message so the caller can fix a typo.
    """
    info = ARCH_REGISTRY.get(name)
    if info is None:
        raise ConfigError(f"Unknown encoder arch {name!r}; registered archs: {list_encoders()}")
    return info


def arch_mean_std(arch: str) -> tuple[float, float]:
    """Return the arch's pretraining ``(mean, std)`` normalisation.

    The normalisation is a frozen property of the *pretrained* encoder (the arch),
    not a user knob — there is no encoder config. Call-time overrides happen via
    the ``norm=`` argument on :meth:`EMEncoder.forward`.
    """
    info = arch_info(arch)
    return info.mean, info.std


# --------------------------------------------------------------------------------------
# Internal test hooks (NOT public API).
# --------------------------------------------------------------------------------------


def register_arch(
    name: str,
    factory: ArchFactory,
    *,
    backbone_attr: str = "vit",
    mean: float = 0.0,
    std: float = 1.0,
    stride: int = 1,
    description: str = "",
    weights_url: str = "",
) -> None:
    """Register a new arch factory. Test-only — duplicates raise.

    Args:
        name: The string a YAML ``arch: ...`` field selects.
        factory: A zero-arg callable returning a built ``nn.Module``.
        mean: The arch's pretraining normalisation mean (default ``0.0`` identity;
            test arches usually leave it identity since random weights don't care).
        std: The arch's pretraining normalisation std (default ``1.0`` identity).
        description: Catalog text (optional; tests usually omit it).
        weights_url: Catalog URL (optional; tests usually omit it).

    Raises:
        ConfigError: If ``name`` is already in :data:`ARCH_REGISTRY`. No silent
            overwrite (mirrors the public registry's duplicate-name rule).
    """
    if name in ARCH_REGISTRY:
        raise ConfigError(f"arch {name!r} is already registered")
    ARCH_REGISTRY[name] = ArchInfo(
        factory=factory,
        backbone_attr=backbone_attr,
        mean=mean,
        std=std,
        stride=stride,
        description=description,
        weights_url=weights_url,
    )


def unregister_arch(name: str) -> None:
    """Remove an arch entry (test-only). Missing entry is a no-op."""
    ARCH_REGISTRY.pop(name, None)


__all__ = [
    "ARCH_REGISTRY",
    "ArchFactory",
    "ArchInfo",
    "arch_info",
    "arch_mean_std",
    "list_encoders",
    "register_arch",
    "unregister_arch",
]

"""``OmniEM`` — the user-facing model wrapper.

Public surface (0.1.1 — channel-less, two-tier):

    OmniEM.load(config, weights=None, *, backbone=None, head=None,
                device=None, dtype=None) -> OmniEM
    OmniEM.from_config(config) -> OmniEM                 # random init, test path
    model.run(image, *, axes, norm=None, conform="strict", squeeze="",
              dtype=None, return_logits=False) -> torch.Tensor   # raw image -> output
    model.predict(tensor) -> torch.Tensor               # canonical [b,z,y,x] -> logits
    model.save_weights(path=None, *, backbone=None, head=None) -> Path | (Path, Path)
    model.config, model.device, model.dtype, model.mean, model.std

There is NO bundle, NO meta, NO key rename, NO tag. The model is built from the
user's :class:`ModelConfig` (the recipe) + raw weights file(s); the only check at
load is :meth:`torch.nn.Module.load_state_dict` ``strict=True``.

Two tiers:

* :meth:`run` — the everyday, safe path: prep → compute → recover from a raw image
  in one call, threading every recovery arg internally (no mismatch possible).
  ``return_logits=False`` gives the task output (needs ``config.task_type``);
  ``return_logits=True`` gives restored caller-layout FLOAT logits.
* :meth:`predict` — the power path: pure compute on a **canonical** ``[b, z, y, x]``
  (ZYX, channel-less, ``z == img_z``) float tensor → canonical logits
  ``[b, c_out, z, y, x]`` (no recovery). Integer tensors raise.

Internals:

* ``self._net`` is the :class:`OmniEMV1Net` (encoder + STAdapter
  z-fusion + UNETR decoder). The backbone lives at ``self._net.vit``.
* The prep / recover stages — ``_apply_input`` (raw → canonical + ``orig_yx``),
  ``_restore`` (logits → caller layout), ``_apply_output`` (task transform) — are
  internal; ``run`` is built from them.
* Normalisation is owned by the MODEL prep: ``norm=None`` resolves to
  ``config.mean`` / ``config.std`` (the FIXED training norm — the head's own
  training statistics, NOT arch-derived, NOT per-image). It is **scalar**.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from omniem._pipeline import channel_insert, parse_squeeze
from omniem.config.model import ModelConfig
from omniem.encoders.base import EMEncoder
from omniem.encoders.dinov2.forward import (
    _PER_IMAGE,
    _PRENORMALIZED,
    _per_image_stats,
    _warn_tensor_out_of_unit_range,
    _warn_values_out_of_unit_range,
)
from omniem.errors import ConfigError, InputContractError, WeightFormatError
from omniem.models.omniemv1_net import OmniEMV1Net
from omniem.models.registry import model_arch_info

ArrayLike = torch.Tensor | np.ndarray

# Valid axes characters — **channel-less** (EM is grayscale; ``in_chans`` is a
# model-internal detail). A ``c`` axis raises (declaring an RGB-stored layout is a
# CLI-only concern: ``--axes cyx`` + ``--color-to-gray`` reduce it *before* the API).
_AXES_VALID: frozenset[str] = frozenset({"b", "z", "y", "x"})

class OmniEM(nn.Module):
    """Inference-only OmniEM model (encoder + STAdapter + UNETR decoder).

    Construct via :meth:`load` (real weights) or :meth:`from_config` (random
    init, for tests). Direct ``__init__`` is supported but not the public path.
    """

    def __init__(
        self,
        net: OmniEMV1Net,
        config: ModelConfig,
        *,
        _own_encoder: bool = True,
    ) -> None:
        super().__init__()
        self._net = net
        self._config = config
        # Whether this model OWNS its encoder. A borrowed (shared-encoder) model
        # sets this False: the inference-only finalisation + every whole-model
        # mutator is then scoped to the head so the shared encoder is never touched.
        self._own_encoder = bool(_own_encoder)
        # Inference-only by contract.
        if self._own_encoder:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False
        else:
            # Borrowed: scope eval + freeze to the head partition; never recurse
            # into the shared encoder subtree (its mode/grad are the caller's).
            self._set_head_training(False)
            enc = self._net.encoder_prefix
            for name, p in self._net.named_parameters():
                if name == enc or name.startswith(enc + "."):
                    continue
                p.requires_grad = False

    # ---- introspection ------------------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:  # pragma: no cover
            return torch.float32

    @property
    def mean(self) -> float:
        """The model's FIXED training normalisation mean (from the config).

        NOT per-image (per-image norm is forbidden here) and NOT
        ``arch_mean_std`` (the encoder arch's stat is a separate fact, used
        by the standalone :class:`EMEncoder`).
        """
        return float(self._config.mean)

    @property
    def std(self) -> float:
        """The model's FIXED training normalisation std (from the config)."""
        return float(self._config.std)

    @property
    def _stride(self) -> int:
        """The arch's input XY divisor (read from the model arch catalog).

        Stride is intrinsic to the arch (omniemv1 → 112), NOT a
        brand-level public property of :class:`OmniEM` and NOT a config field.
        The discoverable accessor is :func:`omniem.model_arch_info`; this
        leading-underscore property is the internal handle the conform /
        un-conform code uses.
        """
        return int(model_arch_info(self._config.arch).stride)

    # ---- load ---------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        config: str | Path | ModelConfig,
        weights: str | Path | None = None,
        *,
        backbone: str | Path | None = None,
        head: str | Path | None = None,
        encoder: EMEncoder | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> OmniEM:
        """Build the model from ``config`` and load weights (with placement).

        Same **optional, separable** loading as :meth:`from_config` — none /
        merged / encoder-only / head-only / both — plus ``device`` / ``dtype``
        placement and the inference-only finalisation:

        * ``weights=`` → a merged whole-model ``state_dict`` (``backbone`` /
          ``head`` then ignored);
        * ``backbone=`` and/or ``head=`` → load each group on its own; a group
          that is **not** supplied keeps its random init;
        * nothing → random init.

        Backbone vs head is partitioned by the encoder's derived prefix
        (:attr:`OmniEMV1Net.encoder_prefix`); each group file must carry exactly
        its group's keys.

        Pass ``encoder=`` (a pre-built :class:`EMEncoder`) to **borrow** a shared
        backbone instead of building one — see :meth:`from_config` for the borrow
        contract. With ``encoder=``, only ``head=`` may accompany it (``weights=`` /
        ``backbone=`` are rejected), and ``device`` / ``dtype`` must match the
        encoder's placement (default: the encoder's).

        Args:
            config: A :class:`ModelConfig`, a path, or an inline YAML string.
            weights: Path to a merged ``state_dict`` ``.pt``.
            backbone: Path to an encoder-backbone-only ``state_dict`` ``.pt``.
            head: Path to a head-only ``state_dict`` ``.pt``.
            encoder: A pre-built :class:`EMEncoder` to borrow by reference (shared
                backbone). Mutually exclusive with ``weights`` / ``backbone``.
            device: Target device (default CPU; with ``encoder=``, the encoder's).
            dtype: Target dtype (default the backbone's native dtype).

        Raises:
            WeightFormatError: Unreadable file, non-tensor value, a group file
                whose keys do not match its group, or a strict-load mismatch.
            ConfigError: Unknown ``config.arch``; or (borrow) encoder arch mismatch
                / head-tensor shape incompatible with the config.
            InputContractError: (borrow) ``encoder`` combined with ``weights`` /
                ``backbone``; a non-inference-ready or mis-placed encoder.
        """
        resolved_cfg = _resolve_model_config(config)
        if encoder is not None:
            return cls._build_borrowed(
                resolved_cfg,
                encoder=encoder,
                head_file=head,
                conflicts={"weights": weights, "backbone": backbone},
                device=device,
                dtype=dtype,
            )
        net = _build_net(resolved_cfg)
        model = cls(net, resolved_cfg)

        # Optional, separable loading (shared with from_config). When `weights`
        # is given the split args are ignored; an un-supplied group keeps its
        # init; nothing supplied → random init.
        _load_weights_into_net(
            net,
            merged=weights,
            encoder=backbone,
            head=head,
            encoder_arg="backbone",
            head_arg="head",
        )

        target_device = torch.device(device) if device is not None else torch.device("cpu")
        try:
            target_dtype = dtype if dtype is not None else next(net.parameters()).dtype
        except StopIteration:  # pragma: no cover
            target_dtype = torch.float32
        if target_dtype == torch.float16 and target_device.type == "cpu":
            raise InputContractError(
                "Cannot load model as float16 on CPU — most CPU conv kernels do "
                "not support fp16. Use device='cuda' or dtype=torch.float32."
            )
        model.to(device=target_device, dtype=target_dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    # ---- construction (optional, separable weight loading) ------------------------

    @classmethod
    def from_config(
        cls,
        config: str | Path | ModelConfig,
        *,
        weights: str | Path | None = None,
        encoder_weights: str | Path | None = None,
        head_weights: str | Path | None = None,
        encoder: EMEncoder | None = None,
    ) -> OmniEM:
        """Build an :class:`OmniEM` from ``config``, **optionally** loading weights.

        Loading is optional and the encoder / head can be loaded **separately**:

        * **no weights** → random init (the test / scaffolding path);
        * ``weights=`` → load a single merged ``state_dict`` (whole model).
          When ``weights`` is given, ``encoder_weights`` / ``head_weights`` are
          **ignored**;
        * ``encoder_weights=`` and/or ``head_weights=`` → load each group on its
          own. A group that is **not** supplied keeps its **random init** — so
          ``encoder_weights=<emdino.pt>`` alone gives a pretrained backbone under
          a freshly-initialised head (the transfer-learning / ``omniem-train``
          setup), and ``head_weights=`` alone replaces only the head.

        Partial loading starts from the freshly-built (random-init) ``state_dict``,
        overlays whichever group(s) are provided, then runs the same full strict
        :func:`torch.nn.Module.load_state_dict` as :meth:`load`. Each group file
        must carry **exactly** its group's keys. The two groups are the **encoder
        backbone** and the **head** (decoder + adapters + ``out``); the backbone
        is whatever sits under :attr:`OmniEMV1Net.encoder_prefix` — a prefix
        **derived from the encoder module**, never a hard-coded name, so a
        different backbone works with no change here.

        The returned model is **inference-only by default** (``eval`` + every
        parameter ``requires_grad=False``), exactly like :meth:`load`. For
        training, call :meth:`prepare_train`. For device / dtype placement, chain
        ``.to(...)`` (or use :meth:`load`, which takes ``device=`` / ``dtype=``).

        Args:
            config: A :class:`ModelConfig`, a path, or an inline YAML string
                (same as :meth:`load`).
            weights: Path to a merged ``state_dict`` ``.pt`` (whole model).
            encoder_weights: Path to an **encoder-backbone-only** ``state_dict``,
                as written by ``save_weights(backbone=...)`` or the split
                ``weights_split/backbone_*.pt`` (the backbone parameter group).
            head_weights: Path to a **head-only** ``state_dict`` (decoder +
                adapters + ``out`` — everything not in the backbone group), as
                written by ``save_weights(head=...)``.
            encoder: A pre-built :class:`EMEncoder` to **borrow** by reference (one
                ViT-L shared across many heads — no copy, no backbone re-load). Only
                ``head_weights`` may accompany it (``weights`` / ``encoder_weights``
                are rejected). The encoder must be inference-ready (eval + frozen);
                the head is placed on the encoder's device/dtype. The resulting model
                is inference-only and rejects whole-model mutators (``to``/``train``/
                ``prepare_train``/…) so the shared encoder is never touched.

        Raises:
            WeightFormatError: An unreadable file, a non-tensor value, a group
                file whose keys do not exactly match its group, or a strict-load
                mismatch (e.g. a shape disagreeing with ``config``).
            ConfigError: (borrow) ``encoder.arch`` mismatch, or a head tensor whose
                shape is incompatible with the borrowed encoder's dims.
            InputContractError: (borrow) ``encoder`` combined with ``weights`` /
                ``encoder_weights``; or a non-inference-ready / multi-device encoder.
        """
        resolved_cfg = _resolve_model_config(config)
        if encoder is not None:
            # Borrow a shared encoder; only head_weights may accompany it.
            return cls._build_borrowed(
                resolved_cfg,
                encoder=encoder,
                head_file=head_weights,
                conflicts={"weights": weights, "encoder_weights": encoder_weights},
                device=None,
                dtype=None,
            )
        net = _build_net(resolved_cfg)
        model = cls(net, resolved_cfg)
        _load_weights_into_net(
            net,
            merged=weights,
            encoder=encoder_weights,
            head=head_weights,
            encoder_arg="encoder_weights",
            head_arg="head_weights",
        )
        return model

    # ---- shared-encoder borrow ----------------------------------------------------

    @classmethod
    def _build_borrowed(
        cls,
        config: ModelConfig,
        *,
        encoder: EMEncoder,
        head_file: str | Path | None,
        conflicts: Mapping[str, str | Path | None],
        device: str | torch.device | None,
        dtype: torch.dtype | None,
    ) -> OmniEM:
        """Build an OmniEM that **borrows** ``encoder`` by reference (shared backbone).

        Shared by :meth:`load` and :meth:`from_config` (which name their
        backbone-providing args differently — ``conflicts`` carries them for clear
        per-constructor rejection). Injects ``encoder.vit`` (no copy / no re-load of
        the backbone), loads only the head group (atomic), places **only** the head
        on the encoder's device+dtype, and returns an inference-only model whose
        whole-model mutators are head-scoped (see the borrowed-model overrides).
        """
        if not isinstance(encoder, EMEncoder):
            raise InputContractError(
                f"encoder= must be an EMEncoder (got {type(encoder).__name__})."
            )
        for arg_name, val in conflicts.items():
            if val is not None:
                raise InputContractError(
                    f"encoder= (borrow) cannot be combined with {arg_name}= — the "
                    f"borrowed encoder already supplies the backbone; pass only the "
                    f"head (load: head=, from_config: head_weights=)."
                )
        if encoder.arch != config.encoder:
            raise ConfigError(
                f"borrowed encoder.arch={encoder.arch!r} != config.encoder="
                f"{config.encoder!r}; the encoder must match the model config."
            )
        _check_encoder_ready(encoder)
        enc_device, enc_dtype = _encoder_placement(encoder)
        if device is not None and not _placement_matches(device, enc_device):
            raise InputContractError(
                f"borrowed encoder is on {enc_device} but device={device!r} was "
                f"requested. omniem places only the head and must not relocate the "
                f"shared encoder — move the encoder first, or omit device=."
            )
        if dtype is not None and dtype != enc_dtype:
            raise InputContractError(
                f"borrowed encoder dtype is {enc_dtype} but dtype={dtype!r} was "
                f"requested. omniem must not recast the shared encoder — cast the "
                f"encoder first, or omit dtype=."
            )

        net = _build_net(config, encoder=encoder._backbone)
        if head_file is not None:
            _load_head_only(net, Path(head_file))
        # Place the head on the encoder's CONCRETE device/dtype (not the requested
        # string) so an index-less request like device="cuda" cannot split the head
        # and encoder across cuda:0 / cuda:1.
        _place_head(net, device=enc_device, dtype=enc_dtype)
        return cls(net, config, _own_encoder=False)

    def _set_head_training(self, mode: bool) -> None:
        """Set the wrapper + every non-encoder ``_net`` submodule's ``training`` flag.

        Used by the borrowed-model finalize / ``eval()`` so the "loaded model is
        eval" contract holds **without** recursing into the shared encoder. The
        encoder partition is identified exactly as the state_dict split does:
        a module whose name is ``encoder_prefix`` or starts with
        ``encoder_prefix + "."`` is skipped.
        """
        self.training = mode
        prefix = self._net.encoder_prefix
        for name, module in self._net.named_modules():
            if name == prefix or name.startswith(prefix + "."):
                continue
            module.training = mode

    # ---- borrowed-model lifecycle guards (reject whole-model mutators) -------------

    def _reject_if_borrowed(self, op: str) -> None:
        if not getattr(self, "_own_encoder", True):
            raise InputContractError(
                f"{op} is not allowed on a borrowed-encoder OmniEM — it would mutate "
                f"the shared encoder and corrupt peer models. Manage the encoder via "
                f"its EMEncoder, and rebuild the model to relocate/recast."
            )

    def _apply(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        # Single funnel for to / cuda / cpu / half / float / double / bfloat16 /
        # type / to_empty — guard it once instead of a per-method list.
        self._reject_if_borrowed("Moving/casting the whole model (.to/.cuda/.half/...)")
        return super()._apply(*args, **kwargs)

    def requires_grad_(self, requires_grad: bool = True) -> OmniEM:  # type: ignore[override]
        self._reject_if_borrowed("requires_grad_()")
        return super().requires_grad_(requires_grad)  # type: ignore[return-value]

    def train(self, mode: bool = True) -> OmniEM:  # type: ignore[override]
        if not getattr(self, "_own_encoder", True):
            if mode:
                raise InputContractError(
                    "train()/.train(True) is not allowed on a borrowed-encoder OmniEM "
                    "(it would put the shared encoder in train mode). Borrowed models "
                    "are inference-only."
                )
            # eval() on a borrowed model: head-scoped, never recurse into the encoder.
            self._set_head_training(False)
            return self
        return super().train(mode)  # type: ignore[return-value]

    def apply(self, fn):  # type: ignore[override]
        # ``apply(fn)`` recurses into children (incl. the shared encoder) BEFORE the
        # root, applying ``fn`` to each — so a per-funnel ``_apply``/``train`` guard
        # cannot stop it. Reject the whole call on a borrowed model.
        self._reject_if_borrowed("apply()")
        return super().apply(fn)

    def load_state_dict(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        # Public ``load_state_dict`` copies into ``_net.vit.*`` (the shared encoder).
        # Reject on a borrowed model; the borrow head-load uses a dedicated in-place
        # copier that never touches the backbone.
        self._reject_if_borrowed("load_state_dict()")
        return super().load_state_dict(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        # ``zero_grad`` iterates ``self.parameters()`` (which include the shared
        # encoder's) and clears each ``.grad`` — a mutation of the shared object.
        self._reject_if_borrowed("zero_grad()")
        return super().zero_grad(set_to_none)

    def set_submodule(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        # ``set_submodule("_net.vit.<...>", m)`` replaces a child INSIDE the shared
        # encoder. Reject on a borrowed model (structural mutation of the encoder).
        self._reject_if_borrowed("set_submodule()")
        return super().set_submodule(*args, **kwargs)

    # ---- save ---------------------------------------------------------------------

    def save_weights(
        self,
        path: str | Path | None = None,
        *,
        backbone: str | Path | None = None,
        head: str | Path | None = None,
    ) -> Path | tuple[Path, Path]:
        """Write the model's weights to disk — merged or split.

        Symmetric with :meth:`load`. The save writes ONLY tensors
        — no meta, no config, no key renaming. Reload requires the user's
        config (the recipe) + the weights file(s).

        Args:
            path: Path to a single merged ``state_dict`` ``.pt`` (merged mode).
            backbone: Path to the encoder ``state_dict`` ``.pt`` — the keys under the
                net's derived encoder prefix (``self._net.encoder_prefix``; ``vit.*``
                for the omniemv1 net) (split mode).
            head: Path to the non-encoder ``state_dict`` ``.pt`` — every key NOT under
                that derived prefix (split mode).

        Returns:
            For merged mode the ``path``; for split mode ``(backbone, head)``.

        Raises:
            InputContractError: Neither / both modes; split mode missing a file; or
                split mode with the SAME path for ``backbone`` and ``head``.
        """
        merged = path is not None
        split = backbone is not None or head is not None
        if merged and split:
            raise InputContractError(
                "save_weights: pass either `path=` (merged) OR `backbone=` + "
                "`head=` (split), not both."
            )
        if not merged and not split:
            raise InputContractError(
                "save_weights: one of `path=` (merged) or `backbone=` + "
                "`head=` (split) is required."
            )
        if split and (backbone is None or head is None):
            raise InputContractError(
                "save_weights split mode requires BOTH `backbone=` and `head=`."
            )
        if split and Path(backbone).resolve() == Path(head).resolve():
            # The split writes backbone then head; identical paths would silently
            # leave a head-only file while returning success (data loss).
            raise InputContractError(
                "save_weights split mode requires DISTINCT `backbone=` and `head=` "
                "paths (the same path would make the head write clobber the backbone)."
            )

        # All save paths emit the net's natural bare keys. For the omniemv1 net these
        # are vit.* (encoder) + encoder1.* / adapters.* / ... (head); the partition
        # below splits by the DERIVED encoder prefix, so `vit.*` is specific to that
        # net, not a universal contract. The OmniEM wrapper's `_net.` prefix is never
        # written to disk — the split files use bare keys.
        full = self._net.state_dict()
        if merged:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(full, out_path)
            return out_path

        # split — partition by the encoder's DERIVED prefix (not hard-coded).
        prefix = self._net.encoder_prefix + "."
        bb_dict: dict[str, torch.Tensor] = {}
        head_dict: dict[str, torch.Tensor] = {}
        for k, v in full.items():
            if k.startswith(prefix):
                bb_dict[k] = v
            else:
                head_dict[k] = v

        bb_path = Path(backbone)
        head_path = Path(head)
        bb_path.parent.mkdir(parents=True, exist_ok=True)
        head_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bb_dict, bb_path)
        torch.save(head_dict, head_path)
        return bb_path, head_path

    # ---- run (merged one-step: prep → compute → recover) -------------------------

    def run(
        self,
        image: ArrayLike,
        *,
        axes: str,
        norm: None | str | Mapping[str, float] = None,
        conform: str = "strict",
        squeeze: str = "",
        dtype: str | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Run the full model pipeline from a raw image — the everyday, safe path.

        ``run`` = ``_apply_input → predict → (_restore | _apply_output)`` with every
        recovery arg (``orig_yx`` / ``conform`` / ``axes``) threaded internally, so a
        mismatch is impossible. The input is **channel-less** grayscale; ``axes`` may
        not contain ``c``.

        Args:
            image: Raw grayscale float input (integer dtypes rejected).
            axes: One-character-per-axis string from ``{b, z, y, x}`` describing
                ``image``. Drives the output layout (the predicted ``c_out`` channel
                is inserted after ``b`` if present, else at the front).
            norm: ``None`` → ``config.mean``/``config.std``; ``'per-image'`` →
                per-sample z-score; ``'prenormalized'`` → skip; ``{'mean': m,
                'std': s}`` → **scalar** override.
            conform: ``'strict'`` (default), ``'pad'`` (reflect/replicate, cropped
                back), or ``'resize'`` (bicubic, resized back).
            squeeze: A subset of ``{b, z}`` — drop the named singleton axis from the
                output (else raise). Default ``""`` mirrors ``axes``.
            dtype: Output integer dtype for the task output (``'uint8'`` default).
                Must be ``None`` when ``return_logits=True`` (logits are not
                quantized).
            return_logits: ``False`` (default) → task output (needs
                ``config.task_type``). ``True`` → restored caller-layout FLOAT logits
                (channels intact, no task transform).

        Returns:
            A :class:`torch.Tensor` — the task output (channel collapsed) or the
            restored caller-layout logits, at the caller's original XY.

        Raises:
            InputContractError: ``return_logits=True`` with a non-None ``dtype``;
                ``return_logits=False`` with no ``config.task_type``; plus the usual
                input-contract violations.
        """
        # Fail-fast on the cheap argument-contract errors BEFORE any (expensive)
        # prep / forward, so a bad flag combo never pays for inference first.
        if not isinstance(return_logits, bool):
            raise InputContractError(
                f"OmniEM.run: `return_logits` must be a bool (got "
                f"{type(return_logits).__name__}: {return_logits!r}); a truthy string "
                f"like \"false\" or an int would silently pick the wrong path."
            )
        if return_logits and dtype is not None:
            raise InputContractError(
                "OmniEM.run: `dtype` must be None when return_logits=True — logits "
                "are continuous floats, not a quantized image. Drop dtype= or set "
                "return_logits=False."
            )
        if not return_logits and self._config.task_type is None:
            raise InputContractError(
                "OmniEM.run(return_logits=False) requires config.task_type "
                "(\"image2image\" or \"image2label\") to pick the output transform. "
                "The model has no task_type — call run(..., return_logits=True) and "
                "postprocess the logits yourself."
            )

        tensor, orig_yx = self._apply_input(image, axes=axes, norm=norm, conform=conform)
        logits = self.predict(tensor)
        if return_logits:
            return self._restore(
                logits, axes=axes, orig_yx=orig_yx, conform=conform, squeeze=squeeze
            )
        return self._apply_output(
            logits,
            axes=axes,
            orig_yx=orig_yx,
            conform=conform,
            squeeze=squeeze,
            dtype=dtype if dtype is not None else "uint8",
        )

    # ---- predict (canonical compute) ---------------------------------------------

    def predict(
        self,
        tensor: torch.Tensor,
        *,
        axes: object = None,
        norm: object = None,
        conform: object = None,
    ) -> torch.Tensor:
        """Pure compute on a **canonical** ``[b, z, y, x]`` tensor → canonical logits.

        The power path: ``tensor`` is a pre-built canonical, channel-less,
        **already-normalised** float tensor — ``z == config.img_z`` (``z=1`` for a
        2D model), square + multiple-of-stride XY. ``predict`` validates the shape
        strictly (no normalization inferred), builds the net input ``[B, 1, Y, X, Z]``
        (preserving YX — never a ``y↔x`` swap), runs the net (it does its own
        ``C==1→in_chans`` repeat), and permutes back to canonical logits
        ``[b, c_out, z, y, x]``. No recovery (use :meth:`run` for caller-shape).

        Args:
            tensor: Canonical ``[b, z, y, x]`` FLOAT tensor — integer tensors raise
                (the package never scales int→float). Auto-moved/cast to the model's
                device/dtype after validation.
            axes / norm / conform: **Removed in 0.1.1.** Present only to detect the
                old ``predict(raw, axes=…)`` call and raise a clear migration error.

        Returns:
            Canonical logits ``[b, c_out, z, y, x]`` on ``self.device``.
        """
        if axes is not None or norm is not None or conform is not None:
            raise InputContractError(
                "OmniEM.predict(x, axes=…/norm=…/conform=…) was removed in 0.1.1. "
                "`predict` now takes only a canonical [b, z, y, x] tensor and returns "
                "canonical logits [b, c_out, z, y, x]. Use `model.run(image, axes=…)` "
                "for a raw image, or build the canonical tensor and call "
                "`model.predict(canonical)`."
            )
        if self.dtype == torch.float16 and self.device.type == "cpu":
            raise InputContractError(
                "OmniEM is float16 on CPU — most CPU conv kernels do not support fp16."
            )
        if not isinstance(tensor, torch.Tensor):
            raise InputContractError(
                f"OmniEM.predict expects a canonical [b, z, y, x] torch.Tensor "
                f"(got {type(tensor).__name__}); use run(image, axes=…) for raw input."
            )
        if not tensor.is_floating_point():
            raise InputContractError(
                f"OmniEM.predict expects a FLOAT canonical tensor (got dtype="
                f"{tensor.dtype}); the package never scales int→float (uint8 != ÷255). "
                f"Use run(image, axes=…) for raw input, or cast first."
            )
        if tensor.ndim != 4:
            raise InputContractError(
                f"OmniEM.predict expects a canonical 4D [b, z, y, x] tensor "
                f"(got ndim={tensor.ndim}, shape={tuple(tensor.shape)}); z=1 for a 2D "
                f"model. Use run(image, axes=…) to canonicalise a raw image."
            )
        B, Z, Y, X = (int(d) for d in tensor.shape)
        if B <= 0 or Z <= 0 or Y <= 0 or X <= 0:
            raise InputContractError(
                f"OmniEM.predict: empty axis (got B={B}, Z={Z}, Y={Y}, X={X})."
            )
        expected_z = int(self._config.img_z)
        if Z != expected_z:
            raise InputContractError(
                f"OmniEM.predict: z must equal config.img_z={expected_z} (got z={Z}); "
                f"flexible Z is not supported."
            )
        stride = self._stride
        if Y != X or Y % stride != 0:
            raise InputContractError(
                f"OmniEM.predict: XY must be square + a multiple of stride={stride} "
                f"(got {Y}x{X}); conform via run(image, axes=…, conform='pad'|'resize')."
            )

        # Move + cast onto the model (after validation).
        if tensor.device != self.device or tensor.dtype != self.dtype:
            tensor = tensor.to(device=self.device, dtype=self.dtype)

        # Build the net input [B, 1, Y, X, Z] — canonical [B, Z, Y, X] → unsqueeze a
        # channel → permute Z to last. YX order is preserved (the net's local X,Y
        # variable names are legacy/internal; never a public y↔x swap). The net does
        # its own C==1→in_chans repeat.
        x5 = tensor.unsqueeze(1).permute(0, 1, 3, 4, 2).contiguous()  # [B, 1, Y, X, Z]
        raw = self._net(x5)  # [B, C_out, Y, X, Z] — PURE LOGITS
        # Permute back to canonical [B, C_out, Z, Y, X].
        return raw.permute(0, 1, 4, 2, 3).contiguous()

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

        Owns the wrapper validation (fp16-CPU, int-dtype, ndarray→tensor), the axes
        canonicalisation to channel-less ``[b, z, y, x]``, the XY conform
        (``strict|pad|resize``), and the **scalar** mean/std affine. Channel
        synthesis is the net's concern (``C==1→in_chans``). ``run`` threads the
        returned ``orig_yx`` into ``_restore`` / ``_apply_output`` (single source of
        truth — never recomputed).
        """
        if self.dtype == torch.float16 and self.device.type == "cpu":
            raise InputContractError(
                "OmniEM is float16 on CPU — most CPU conv kernels do not support fp16."
            )
        x = self._coerce_input(image)
        cleaned = _parse_axes(axes)
        if x.ndim != len(cleaned):
            raise InputContractError(
                f"image.ndim ({x.ndim}) does not match axes={cleaned!r} "
                f"(length {len(cleaned)})"
            )

        # Reorder to canonical [B, Z, Y, X], adding singletons for missing axes.
        order = ["b", "z", "y", "x"]
        present = set(cleaned)
        perm = [cleaned.index(ax) for ax in order if ax in present]
        x = x.permute(*perm) if perm else x
        pos = 0
        for ax in order:
            if ax not in present:
                x = x.unsqueeze(pos)
            pos += 1

        B, Z, Y, X = x.shape
        orig_yx = (int(Y), int(X))
        if B <= 0 or Z <= 0 or Y <= 0 or X <= 0:
            raise InputContractError(
                f"OmniEM.apply_input: empty spatial / batch axis (got B={B}, Z={Z}, "
                f"Y={Y}, X={X}); inputs must have positive size."
            )
        expected_z = int(self._config.img_z)
        if Z != expected_z:
            raise InputContractError(
                f"Z must equal config.img_z={expected_z}; got Z={Z}. "
                f"Flexible/same-padded Z is not supported."
            )

        # Per-image: capture per-sample stats on the PRE-CONFORM input.
        per_image = isinstance(norm, str) and norm == _PER_IMAGE
        pi_mean: torch.Tensor | None = None
        pi_std: torch.Tensor | None = None
        if per_image:
            pi_mean, pi_std = _per_image_stats(x.to(torch.float32))
        elif norm != _PRENORMALIZED:
            _warn_tensor_out_of_unit_range(x.to(torch.float32), what="OmniEM input")

        stride = self._stride
        if conform == "strict":
            if Y != X:
                raise InputContractError(
                    f"XY must be square (got Y={Y}, X={X}); EM is in-plane isotropic."
                )
            if Y % stride != 0:
                raise InputContractError(
                    f"XY side must be a multiple of stride={stride}; got {Y}."
                )
        elif conform == "pad":
            x = _conform_pad_xy(x, stride=stride)
        elif conform == "resize":
            x = _conform_resize_xy(x, stride=stride)
        else:
            raise InputContractError(
                f"conform must be one of 'pad', 'resize', 'strict' (got {conform!r})"
            )

        # Float cast for the normalisation arithmetic.
        x = x.to(torch.float32)

        if per_image:
            assert pi_mean is not None and pi_std is not None
            m = pi_mean.to(device=x.device).view(B, 1, 1, 1)
            s = pi_std.to(device=x.device).view(B, 1, 1, 1)
            x = ((x - m) / s).to(x.dtype)
        else:
            x = _apply_norm(x, config=self._config, norm=norm)

        return x, orig_yx

    # ---- internal recover stages -------------------------------------------------

    def _restore(
        self,
        logits: torch.Tensor,
        *,
        axes: str,
        orig_yx: tuple[int, int],
        conform: str,
        squeeze: str = "",
    ) -> torch.Tensor:
        """Un-conform XY + reshape canonical logits → caller ``axes`` (+ ``squeeze``).

        ``logits`` is canonical ``[B, C_out, Z, Y, X]`` (from :meth:`predict`). The
        un-conform runs on the **continuous logits** before any task transform
        (caller path: :meth:`run` with ``return_logits=True``). Output channel is
        inserted per :func:`omniem._pipeline.channel_insert`.
        """
        cleaned = _parse_axes(axes)
        drop = parse_squeeze(squeeze)
        restored, labels = self._unconform_and_layout(
            logits, cleaned=cleaned, orig_yx=orig_yx, conform=conform
        )
        restored, _ = _drop_squeeze(restored, labels, drop)
        return restored

    def _apply_output(
        self,
        logits: torch.Tensor,
        *,
        axes: str,
        orig_yx: tuple[int, int],
        conform: str,
        squeeze: str,
        dtype: str,
    ) -> torch.Tensor:
        """Model-owned output stage — un-conform → task transform → collapse → squeeze.

        Runs ``_restore``'s un-conform + caller layout (keeping ``c``), then the
        ``task_type``-gated transform (``image2image``: sigmoid+quantize;
        ``image2label``: argmax — both **collapse the channel axis**), then applies
        ``squeeze`` (exactly once). Requires ``config.task_type``.
        """
        from omniem.models import output as _output  # local — avoids import cycle

        task_type = self._config.task_type
        if task_type is None:
            raise InputContractError(
                "model output stage requires config.task_type "
                "(\"image2image\" or \"image2label\"); the model has no task_type so "
                "it cannot decide the output transform — use run(..., "
                "return_logits=True) and postprocess the logits yourself."
            )
        _output._resolve_dtype(dtype)  # surface an unknown dtype early

        cleaned = _parse_axes(axes)
        drop = parse_squeeze(squeeze)
        restored, labels = self._unconform_and_layout(
            logits, cleaned=cleaned, orig_yx=orig_yx, conform=conform
        )
        ch_axis = labels.index("c")
        actual_c = int(restored.shape[ch_axis])
        expected_c = int(self._config.out_channels)
        if actual_c != expected_c:
            raise InputContractError(
                f"output stage (axes={axes!r}): logits channel size {actual_c} "
                f"(dim {ch_axis}) disagrees with config.out_channels={expected_c}."
            )

        if task_type == "image2image":
            if expected_c != 1:
                raise InputContractError(
                    f"task_type='image2image' implies out_channels==1; got {expected_c}"
                )
            out = _output._apply_image2image(restored, ch_axis=ch_axis, dtype=dtype)
        elif task_type == "image2label":
            full_scale, _ = _output._resolve_dtype(dtype)
            if expected_c - 1 > full_scale:
                raise InputContractError(
                    f"task_type='image2label' with out_channels={expected_c} does not "
                    f"fit in dtype={dtype!r} (max class id {expected_c - 1} > "
                    f"{full_scale}); use 'uint16'."
                )
            out = _output._apply_image2label(restored, ch_axis=ch_axis, dtype=dtype)
        else:  # pragma: no cover — Literal already constrains task_type
            raise InputContractError(f"Unknown task_type={task_type!r}")

        # The channel axis is gone; the remaining labels keep their order.
        labels_after = [a for a in labels if a != "c"]
        out, _ = _drop_squeeze(out, labels_after, drop)
        return out

    # ---- migration stubs (removed public surface) --------------------------------

    def apply_input(self, *args: Any, **kwargs: Any):
        """Removed in 0.1.1 — use :meth:`run` / build a canonical for :meth:`predict`."""
        raise InputContractError(
            "OmniEM.apply_input was removed in 0.1.1 (no more Prepared carrier). "
            "Use `model.run(image, axes=…)` for the full pipeline, or build a "
            "canonical [b, z, y, x] tensor and call `model.predict(canonical)`."
        )

    def apply_output(self, *args: Any, **kwargs: Any):
        """Removed in 0.1.1 — :meth:`run` owns the output stage."""
        raise InputContractError(
            "OmniEM.apply_output was removed in 0.1.1. Use "
            "`model.run(image, axes=…)` for the task output (or "
            "`run(..., return_logits=True)` for caller-layout logits)."
        )

    # ---- training handoff ---------------------------------------------------------

    def prepare_train(self, *, fix_encoder: bool = True) -> OmniEM:
        """Flip the (inference-only-by-default) model into a trainable state.

        :meth:`load` / :meth:`from_config` ship the model **frozen + eval** (the
        package's inference contract: every parameter ``requires_grad=False`` and
        ``self.eval()``). A training consumer (e.g. ``omniem-train``) calls this
        once to hand the model over to a ``torch`` optimiser:

        * puts the whole model in **train mode** (``self.train()`` — re-enables
          dropout / drop-path / norm-stat updates the inference default disabled);
        * sets ``requires_grad=True`` on **every** parameter, then —
        * when ``fix_encoder`` (default) — sets ``requires_grad=False`` on the
          **encoder backbone** so its weights are not learnable. The **head**
          (UNETR decoder + ``out``) and the **STAdapters** (which are NOT part of
          the backbone) stay trainable — the adapter-tuning / frozen-backbone
          setup.

        The encoder backbone is identified by :attr:`OmniEMV1Net.encoder_prefix` —
        a prefix **derived from the encoder module itself**, the same definition
        the split save/load uses. A backbone stored under a different attribute
        name is frozen correctly with no change here.

        Freezing toggles ``requires_grad`` on the backbone only and leaves it in
        **train mode** (drop-path stays active — matching how the shipped heads
        were trained). To run the frozen backbone deterministically instead, set
        the backbone submodule to ``eval()`` after this.

        Args:
            fix_encoder: When ``True`` (default), freeze the encoder backbone —
                its parameters are not learnable. When ``False``, the whole model
                (encoder + head + adapters) is trainable.

        Returns:
            ``self`` (chainable, e.g. ``model.prepare_train().to('cuda')``).

        Raises:
            InputContractError: On a borrowed-encoder model — training would mutate
                the shared encoder; rebuild an owned model to train.
        """
        self._reject_if_borrowed("prepare_train()")
        self.train()
        for p in self.parameters():
            p.requires_grad = True
        if fix_encoder:
            # The encoder = the backbone partition, identified by the encoder's
            # DERIVED prefix (OmniEMV1Net.encoder_prefix) — NOT a hard-coded `vit`.
            # The STAdapters / UNETR head are outside this prefix and stay
            # trainable. Iterate _net's OWN names (no `_net.` wrapper prefix).
            prefix = self._net.encoder_prefix + "."
            for name, p in self._net.named_parameters():
                if name.startswith(prefix):
                    p.requires_grad = False
        return self

    # ---- internals ----------------------------------------------------------------

    # ---- coercion + prepared build (owned by apply_input/predict) ----------------

    def _coerce_input(self, x: ArrayLike) -> torch.Tensor:
        """Reject int dtypes / wrong types — owned by apply_input + predict.

        Pulled out of the old ``predict`` body so :meth:`apply_input` and the
        raw-input branch of :meth:`predict` share exactly one validator
        Does NOT move to ``self.device`` here — the model
        still moves the prepared tensor onto its device just before the net
        forward (the prep is shape-arithmetic only).
        """
        if isinstance(x, np.ndarray):
            if not np.issubdtype(x.dtype, np.floating):
                raise InputContractError(
                    f"OmniEM requires a float ndarray (got dtype={x.dtype}); "
                    f"the package does not guess int→float scaling."
                )
            return torch.from_numpy(np.ascontiguousarray(x))
        if isinstance(x, torch.Tensor):
            if not x.is_floating_point():
                raise InputContractError(
                    f"OmniEM requires a floating-point tensor "
                    f"(got dtype={x.dtype}); the package does not guess "
                    f"int→float scaling (uint8 != ÷255). Cast on the caller "
                    f"side, e.g. ``x.float() / 255``."
                )
            return x
        raise InputContractError(
            f"OmniEM x must be torch.Tensor or numpy.ndarray "
            f"(got {type(x).__name__})"
        )

    def _unconform_and_layout(
        self,
        logits: torch.Tensor,
        *,
        cleaned: str,
        orig_yx: tuple[int, int],
        conform: str,
    ) -> tuple[torch.Tensor, list[str]]:
        """Un-conform XY then reshape canonical logits to the caller layout.

        ``logits`` is canonical ``[B, C_out, Z, Y, X]``. Returns ``(tensor, labels)``
        where ``labels`` is the ordered axis list of the output (still **carrying
        ``c``**; the task transform / squeeze run afterwards). Rule 1 (axes drives
        layout) + rule 2 (never drop a non-singleton ``b``/``z``) are enforced here.
        """
        # Validate the logits tensor itself (type / rank / float) before reading its
        # XY dims — a bad internal call surfaces a clear InputContractError rather than
        # a raw IndexError / unpack failure. ``predict`` produces a 5D float canonical;
        # this guards direct ``_restore`` / ``_apply_output`` callers (incl. tests).
        if not isinstance(logits, torch.Tensor):
            raise InputContractError(
                f"logits must be a torch.Tensor (got {type(logits).__name__})."
            )
        if logits.ndim != 5:
            raise InputContractError(
                f"logits must be canonical 5D [B, C, Z, Y, X] (got ndim={logits.ndim}, "
                f"shape={tuple(logits.shape)})."
            )
        if not logits.is_floating_point():
            raise InputContractError(
                f"logits must be a FLOAT tensor (got dtype={logits.dtype}); "
                f"predict returns float logits."
            )
        oy, ox = orig_yx
        if not (isinstance(oy, int) and isinstance(ox, int) and oy > 0 and ox > 0):
            raise InputContractError(
                f"orig_yx must be positive ints (got {orig_yx!r})."
            )
        # logits canonical [B, C, Z, Y, X] — Y,X are the last two dims.
        Hc, Wc = int(logits.shape[3]), int(logits.shape[4])
        if conform == "strict":
            if (oy, ox) != (Hc, Wc):
                raise InputContractError(
                    f"orig_yx={orig_yx} but conform='strict' and the logits XY is "
                    f"{Hc}x{Wc} (strict is a no-op round-trip)."
                )
        elif conform == "pad":
            if oy > Hc or ox > Wc:
                raise InputContractError(
                    f"orig_yx={orig_yx} exceeds the conformed XY ({Hc}x{Wc}) — cannot "
                    f"crop a pad round-trip back to a larger size."
                )
            logits = logits[..., :oy, :ox]
        elif conform == "resize":
            logits = self._resize_logits_xy(logits, (oy, ox))
        else:
            raise InputContractError(
                f"_restore: unknown conform mode {conform!r}."
            )

        # Reshape canonical [B, C, Z, Y, X] → caller layout (channel inserted per the
        # shared rule). Drop b/z the caller did not name (rule 2: must be singleton).
        return self._to_caller_axes(logits, cleaned=cleaned)

    def _to_caller_axes(
        self,
        logits: torch.Tensor,
        *,
        cleaned: str,
    ) -> tuple[torch.Tensor, list[str]]:
        """Reshape canonical ``[B, C_out, Z, Y, X]`` logits to caller-axes order.

        Output mirrors the caller's (channel-less) ``axes`` with ``c_out`` inserted
        per :func:`omniem._pipeline.channel_insert` (after ``b`` if present, else
        front). ``b``/``z`` the caller did not name are dropped — but only when
        singleton (rule 2). Returns ``(tensor, labels)``.

        Examples:
            * ``axes='yx'``  → ``[C, Y, X]``       (B,Z dropped — must be singleton)
            * ``axes='zyx'`` → ``[C, Z, Y, X]``
            * ``axes='byx'`` → ``[B, C, Y, X]``
            * ``axes='bzyx'``→ ``[B, C, Z, Y, X]``
            * ``axes='byxz'``→ ``[B, C, Y, X, Z]``  (downstream [B,C,Y,X,Z] path)
        """
        has_b = "b" in cleaned
        has_z = "z" in cleaned
        B = int(logits.shape[0])
        Z = int(logits.shape[2])
        if not has_b and B != 1:
            raise InputContractError(
                f"axes={cleaned!r} has no 'b' but the batch dim is {B} > 1 — the "
                f"output layout would silently drop all but the first. Add 'b' to "
                f"axes or run one tile at a time."
            )
        if not has_z and Z != 1:
            raise InputContractError(
                f"axes={cleaned!r} has no 'z' but the depth dim is {Z} > 1 — the "
                f"output layout would silently drop the z axis. Add 'z' to axes."
            )
        internal = ["b", "c", "z", "y", "x"]
        # Drop z (dim 2) then b (dim 0) for axes the caller omitted (singleton-safe).
        if not has_z:
            logits = logits.squeeze(2)
            internal.remove("z")
        if not has_b:
            logits = logits.squeeze(0)
            internal.remove("b")
        spatial_order = [ax for ax in cleaned if ax in ("y", "x", "z") and ax in internal]
        target = (["b"] if has_b else []) + spatial_order
        # The predicted c_out is inserted per the shared channel_insert rule (after
        # b if present, else front) — the single source of truth.
        target.insert(channel_insert(cleaned), "c")
        out = _reorder_axes(logits, internal=internal, target=target)
        return out, target

    def _resize_logits_xy(
        self,
        logits: torch.Tensor,
        orig_yx: tuple[int, int],
    ) -> torch.Tensor:
        """Bicubic-resize the canonical ``[B, C, Z, Y, X]`` logits XY back to ``orig_yx``.

        Folds ``(B, Z) → B*Z`` so the 4D interpolate touches only the last two
        (Y, X) axes, then un-folds. Z is never resized.
        """
        B, C, Z, Hp, Wp = logits.shape
        oy, ox = int(orig_yx[0]), int(orig_yx[1])
        x4 = (
            logits.permute(0, 2, 1, 3, 4)
            .reshape(B * Z, C, Hp, Wp)
            .to(dtype=torch.float32)
        )
        x4 = torch.nn.functional.interpolate(
            x4, size=(oy, ox), mode="bicubic", align_corners=False
        )
        return (
            x4.reshape(B, Z, C, oy, ox)
            .permute(0, 2, 1, 3, 4)
            .to(dtype=logits.dtype)
        )


# --------------------------------------------------------------------------------------
# Module-level helpers.
# --------------------------------------------------------------------------------------


def _check_encoder_ready(encoder: EMEncoder) -> None:
    """Reject a borrowed encoder that is not inference-ready.

    Deterministic inference holds only for a fully frozen / eval encoder. A caller
    can ``encoder.vit.train()`` (or unfreeze) while the wrapper's ``training`` flag
    stays False, so check **every** submodule's mode, not just ``encoder.training``,
    and reject any trainable parameter.
    """
    for module in encoder.modules():
        if module.training:
            raise InputContractError(
                "borrowed encoder must be in eval mode — a submodule has "
                "training=True (call encoder.eval() before borrowing)."
            )
    for p in encoder.parameters():
        if p.requires_grad:
            raise InputContractError(
                "borrowed encoder must be frozen — a parameter has "
                "requires_grad=True (omniem must not change the shared encoder)."
            )


def _placement_matches(requested: str | torch.device, actual: torch.device) -> bool:
    """True if ``requested`` refers to the same device as ``actual``.

    An index-less device of the same type (e.g. ``"cuda"`` vs ``cuda:0``, ``"cpu"``
    vs ``cpu``) is treated as a match, so an explicit ``device="cuda"`` is not
    falsely rejected for an encoder on ``cuda:0``. The head is still physically
    placed on the encoder's concrete device, never on the index-less request.
    """
    req = torch.device(requested)
    if req.type != actual.type:
        return False
    if req.index is None or actual.index is None:
        return True
    return req.index == actual.index


def _encoder_placement(encoder: EMEncoder) -> tuple[torch.device, torch.dtype]:
    """Return the borrowed encoder's single ``(device, float-dtype)``, validated.

    Device must be identical across **all** params + buffers; the floating-point
    dtype must be identical across all floating params/buffers (integer/bool buffers
    are legitimate and ignored). Keeps the fp16-on-CPU guard for the borrowed path.
    """
    devices: set[torch.device] = set()
    fdtypes: set[torch.dtype] = set()
    for t in itertools.chain(encoder.parameters(), encoder.buffers()):
        devices.add(t.device)
        if t.is_floating_point():
            fdtypes.add(t.dtype)
    if len(devices) != 1:
        raise InputContractError(
            f"borrowed encoder spans multiple devices {sorted(map(str, devices))}; "
            f"place it on one device before borrowing."
        )
    if len(fdtypes) != 1:
        raise InputContractError(
            f"borrowed encoder spans multiple float dtypes {sorted(map(str, fdtypes))}; "
            f"cast it to one dtype before borrowing."
        )
    device = next(iter(devices))
    dtype = next(iter(fdtypes))
    if dtype == torch.float16 and device.type == "cpu":
        raise InputContractError(
            "borrowed encoder is float16 on CPU — most CPU conv kernels do not "
            "support fp16. Use a CUDA device or float32."
        )
    return device, dtype


def _head_keys(net: OmniEMV1Net) -> set[str]:
    """The non-encoder (head) ``state_dict`` keys of ``net`` (by ``encoder_prefix``)."""
    prefix = net.encoder_prefix + "."
    return {k for k in net.state_dict() if not k.startswith(prefix)}


def _load_head_only(net: OmniEMV1Net, path: Path) -> None:
    """Atomically load ONLY the head group into ``net`` (the shared backbone untouched).

    Steps (all-or-nothing — never partially mutate the head, never touch ``vit.*``):

    1. exact head-key check — the file must carry **exactly** the head keys (keys ≠
       head group → :class:`WeightFormatError`);
    2. **preflight every head tensor's shape** against the built head BEFORE any copy
       (a per-tensor ``copy_`` could mutate earlier head tensors then fail mid-way);
       a shape mismatch under matching keys means the encoder's dims are incompatible
       with this head/config → :class:`ConfigError`;
    3. only then copy each head tensor in place. ``load_state_dict`` is never run
       over the backbone.
    """
    file_sd = _validate_tensor_state_dict(_load_raw_state_dict(path), path)
    head_keys = _head_keys(net)

    got = set(file_sd)
    if got != head_keys:
        missing = sorted(head_keys - got)
        extra = sorted(got - head_keys)
        raise WeightFormatError(
            f"{path}: keys do not match the model's head group. "
            f"missing={missing[:5]} extra={extra[:5]} "
            f"(expected exactly the {len(head_keys)} head keys — pass a head-only "
            f"weight file)."
        )

    # Live tensors (params + persistent buffers) keyed by their state_dict names.
    live: dict[str, torch.Tensor] = dict(net.named_parameters())
    live.update(net.named_buffers())

    # (2) atomic shape preflight — raise before any copy.
    for k in head_keys:
        want = live[k].shape
        have = file_sd[k].shape
        if tuple(have) != tuple(want):
            raise ConfigError(
                f"{path}: head tensor {k!r} shape {tuple(have)} != model {tuple(want)} "
                f"— the borrowed encoder's dims are incompatible with this head/config."
            )

    # (3) copy only the head tensors, in place.
    with torch.no_grad():
        for k in head_keys:
            dst = live[k]
            dst.copy_(file_sd[k].to(device=dst.device, dtype=dst.dtype))


def _place_head(net: OmniEMV1Net, *, device: torch.device, dtype: torch.dtype) -> None:
    """Move ONLY the head child modules of ``net`` to ``device``/``dtype``.

    Skips the encoder child (``encoder_prefix``) so the shared backbone is never
    relocated/recast. ``.to(dtype=...)`` casts only floating-point tensors, so
    integer/bool head buffers are preserved.
    """
    prefix = net.encoder_prefix
    for name, child in net.named_children():
        if name == prefix:
            continue
        child.to(device=device, dtype=dtype)


def _build_net(config: ModelConfig, *, encoder: nn.Module | None = None) -> OmniEMV1Net:
    """Build an :class:`OmniEMV1Net` by dispatching ``config.arch`` through the registry.

    When ``encoder`` is given (the shared-encoder borrow path), it is injected into
    the net's factory so the pre-built backbone is reused instead of a fresh build.
    ``encoder`` is passed to the factory **only when borrowing** so that one-arg
    factories on the owned path (and any minimal test factory) stay valid.
    """
    info = model_arch_info(config.arch)  # raises ConfigError on unknown arch
    if encoder is not None:
        return info.factory(config, encoder=encoder)
    return info.factory(config)


def _resolve_model_config(config: str | Path | ModelConfig) -> ModelConfig:
    """Decode the ``config`` argument to a :class:`ModelConfig`."""
    if isinstance(config, ModelConfig):
        return config
    if isinstance(config, Path):
        if not config.is_file():
            raise ConfigError(f"Model config file not found: {str(config)!r}")
        return ModelConfig.from_yaml(config)  # type: ignore[return-value]
    if isinstance(config, str):
        lower = config.lower()
        if lower.endswith(".yaml") or lower.endswith(".yml"):
            if not Path(config).is_file():
                raise ConfigError(f"Model config file not found: {config!r}")
        return ModelConfig.from_yaml(config)  # type: ignore[return-value]
    raise ConfigError(f"`config` must be str, Path, or ModelConfig (got {type(config).__name__})")


def _parse_axes(axes: str) -> str:
    if not isinstance(axes, str) or not axes:
        raise InputContractError(f"`axes` must be a non-empty string (got {axes!r})")
    cleaned = "".join(axes.split())
    seen: set[str] = set()
    for ax in cleaned:
        if ax == "c":
            # Channel-less contract: the model is grayscale-in; `in_chans` is a
            # model-internal detail. Declaring an RGB-stored layout with `c` is a
            # CLI-only concern (`--axes cyx` + `--color-to-gray`).
            raise InputContractError(
                f"`axes` must not contain 'c' — OmniEM takes grayscale input only "
                f"(channel-less); the model synthesises `in_chans` itself. See "
                f"docs/input-format.md (got axes={axes!r})."
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


def _drop_squeeze(
    tensor: torch.Tensor,
    labels: list[str],
    drop: frozenset[str],
) -> tuple[torch.Tensor, list[str]]:
    """Drop the ``squeeze``-named ``b``/``z`` axes from ``tensor`` (singleton-only).

    ``labels`` is the ordered axis list of ``tensor``. Each axis in ``drop`` must be
    present and singleton (else raise). ``z`` is removed before ``b`` so the
    remaining indices stay valid (``b`` is leftmost). Returns ``(tensor, labels)``.
    """
    labels = list(labels)
    for ax in ("z", "b"):
        if ax not in drop:
            continue
        if ax not in labels:
            raise InputContractError(
                f"squeeze={ax!r} but the output has no {ax!r} axis (it was not in "
                f"`axes`, or was already collapsed)."
            )
        idx = labels.index(ax)
        if int(tensor.shape[idx]) != 1:
            raise InputContractError(
                f"squeeze={ax!r} requires a singleton axis (got size "
                f"{int(tensor.shape[idx])}); a non-singleton {ax!r} cannot be squeezed."
            )
        tensor = tensor.squeeze(idx)
        labels.remove(ax)
    return tensor, labels


def _reorder_axes(
    tensor: torch.Tensor,
    *,
    internal: list[str],
    target: list[str],
) -> torch.Tensor:
    """Permute ``tensor`` from its ``internal`` axis order to ``target`` order."""
    assert sorted(internal) == sorted(target), (
        f"_reorder_axes axis-set mismatch: internal={internal}, target={target}"
    )
    if internal == target:
        return tensor
    perm = [internal.index(ax) for ax in target]
    return tensor.permute(*perm).contiguous()


def _apply_norm(
    x: torch.Tensor,
    *,
    config: ModelConfig,
    norm: None | str | Mapping[str, float],
) -> torch.Tensor:
    """Apply **scalar** ``(x − mean) / std`` per the ``norm=`` directive.

    Operates on the channel-less ``[B, Z, Y, X]`` grid. Handles the
    fixed/override/skip cases. ``'per-image'`` is NOT handled here — it needs the
    pre-conform tensor and is applied by :meth:`_apply_input` upstream.

    * ``norm is None`` → use ``config.mean`` / ``config.std`` (the model's
      FIXED training normalisation).
    * ``norm == 'prenormalized'`` → skip the affine (caller already did it).
    * ``norm`` is a mapping with EXACT keys ``{'mean', 'std'}`` → **scalar**
      override (per-channel sequences are rejected — EM is grayscale).
    """
    if isinstance(norm, str):
        if norm == _PRENORMALIZED:
            return x
        # 'per-image' is valid at the public API but is handled in _apply_input
        # (it needs the pre-conform tensor); it never reaches here.
        raise InputContractError(
            f"norm={norm!r} unknown; allowed: None, '{_PRENORMALIZED}', "
            f"'{_PER_IMAGE}', {{'mean': m, 'std': s}}."
        )
    if norm is None:
        eff_mean = config.mean
        eff_std = config.std
    elif isinstance(norm, Mapping):
        keys = set(norm.keys())
        if keys != {"mean", "std"}:
            raise InputContractError(
                f"norm dict must have EXACT keys {{'mean', 'std'}}; got {sorted(keys)}."
            )
        eff_mean = norm["mean"]
        eff_std = norm["std"]
        # argument-decide → warn (never raise) on out-of-[0,1] override stats.
        _warn_values_out_of_unit_range(eff_mean, what="norm mean")
        _warn_values_out_of_unit_range(eff_std, what="norm std")
    else:
        raise InputContractError(
            f"norm must be None, 'prenormalized', or a dict; got {type(norm).__name__}."
        )

    m_t = _to_scalar_param(eff_mean, device=x.device, name="mean")
    s_t = _to_scalar_param(eff_std, device=x.device, name="std", positive=True)
    return (x - m_t) / s_t


def _to_scalar_param(
    v: float,
    *,
    device: torch.device,
    name: str = "mean/std",
    positive: bool = False,
) -> torch.Tensor:
    """Validate a **scalar** mean-or-std → 0-d tensor. Per-channel sequences rejected."""
    if isinstance(v, bool):
        raise InputContractError(f"{name} must be numeric (got bool)")
    if isinstance(v, (int, float)):
        f = float(v)
        if not math.isfinite(f):
            raise InputContractError(f"{name} must be finite (got {v!r})")
        if positive and f <= 0:
            raise InputContractError(f"{name} must be strictly positive (got {v!r})")
        return torch.tensor(f, device=device, dtype=torch.float32)
    if isinstance(v, (str, bytes, bytearray)):
        raise InputContractError(f"{name} must be numeric, not a string (got {v!r})")
    if isinstance(v, Sequence):
        raise InputContractError(
            f"{name} must be a SCALAR (got a sequence {v!r}); per-channel mean/std is "
            f"rejected — EM is grayscale, the synthesised channels are identical."
        )
    raise InputContractError(
        f"{name} must be a numeric scalar (got {type(v).__name__})"
    )


def _load_raw_state_dict(path: Path) -> dict[str, Any]:
    """``torch.load(weights_only=True)`` + basic shape validation."""
    try:
        raw = torch.load(path, weights_only=True, map_location="cpu")
    except FileNotFoundError:
        raise
    except Exception as e:
        raise WeightFormatError(
            f"{path}: cannot read weights file "
            f"(corrupt, not a torch checkpoint, or wrong format): {e}"
        ) from e
    if not isinstance(raw, dict):
        raise WeightFormatError(
            f"{path}: weights must be a state_dict (got {type(raw).__name__})"
        )
    return raw


def _validate_tensor_state_dict(state: dict[str, Any], path: Path) -> dict[str, torch.Tensor]:
    """Validate a loaded ``state_dict`` is a plain ``str -> Tensor`` mapping.

    omniem loads **clean** state_dicts (its own save format). A file carrying
    non-tensor entries (e.g. a foreign tool's tag / meta fields) is rejected
    here with a clear error instead of an opaque strict-load failure. Adapting a
    foreign file to omniem's format (dropping such entries) is the caller's job,
    not the package's.
    """
    for k, v in state.items():
        if not isinstance(k, str):
            raise WeightFormatError(f"{path}: state_dict has a non-string key {k!r}")
        if not isinstance(v, torch.Tensor):
            raise WeightFormatError(
                f"{path}: value for key {k!r} is {type(v).__name__}, not a Tensor — "
                f"omniem loads clean state_dicts; strip foreign meta entries first."
            )
    return state


def _strict_load_net(net: OmniEMV1Net, state_dict: dict[str, Any]) -> None:
    """Strict ``load_state_dict`` into the bare ``net`` with a clear error.

    The on-disk format uses bare keys (``vit.*`` / ``encoder1.*`` / ``adapters.*``
    / …), not the ``OmniEM`` wrapper's ``_net.`` prefix, so this targets the
    inner module directly. Shared by :meth:`OmniEM.load` and
    :meth:`OmniEM.from_config` (no duplicate strict-load plumbing).
    """
    try:
        result = net.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise WeightFormatError(f"strict load_state_dict failed: {e}") from e
    if result.missing_keys or result.unexpected_keys:
        raise WeightFormatError(
            f"strict load_state_dict left unmatched keys "
            f"(missing={result.missing_keys[:5]}, "
            f"unexpected={result.unexpected_keys[:5]})"
        )


def _overlay_group(
    target: dict[str, torch.Tensor],
    file_sd: dict[str, torch.Tensor],
    expected_keys: set[str],
    *,
    arg_name: str,
    group_desc: str,
    path: Path,
) -> None:
    """Overlay a partial-load weight file onto ``target`` after an exact-key check.

    Used by :meth:`OmniEM.from_config` partial loading: the encoder /head file
    must carry EXACTLY the keys of its group (``expected_keys``) — no missing,
    no extra — so the strict load that follows is fully covered and the caller
    can't silently load the wrong file (e.g. a merged file as ``encoder_weights``).
    """
    got = set(file_sd)
    if got != expected_keys:
        missing = sorted(expected_keys - got)
        extra = sorted(got - expected_keys)
        raise WeightFormatError(
            f"{arg_name}={str(path)!r}: keys do not match the model's {group_desc} "
            f"group. missing={missing[:5]} extra={extra[:5]} "
            f"(expected exactly the {len(expected_keys)} {group_desc} keys; "
            f"pass a {group_desc} weight file, or use weights= for a merged file)."
        )
    target.update(file_sd)


def _load_weights_into_net(
    net: OmniEMV1Net,
    *,
    merged: str | Path | None,
    encoder: str | Path | None,
    head: str | Path | None,
    encoder_arg: str,
    head_arg: str,
) -> None:
    """Optional, separable weight loading into a bare ``net`` (shared core).

    Backs both :meth:`OmniEM.load` and :meth:`OmniEM.from_config`, so loading is
    uniform across both:

    * **nothing supplied** → no-op (the ``net`` keeps its current/random init);
    * ``merged`` → load a whole-model ``state_dict`` (``encoder``/``head`` then
      **ignored**);
    * otherwise → overlay whichever of ``encoder`` / ``head`` is given onto the
      ``net``'s current state (an un-supplied group keeps its init), then run a
      full strict load.

    The backbone group is the encoder's derived prefix
    (:attr:`OmniEMV1Net.encoder_prefix`); each supplied group file must carry
    **exactly** its group's keys. ``encoder_arg`` / ``head_arg`` name the caller's
    kwargs for clear error messages (``encoder_weights`` vs ``backbone``, etc.).
    """
    if merged is None and encoder is None and head is None:
        return
    if merged is not None:
        sd = _validate_tensor_state_dict(_load_raw_state_dict(Path(merged)), Path(merged))
        _strict_load_net(net, sd)
        return

    full = dict(net.state_dict())
    prefix = net.encoder_prefix + "."
    backbone_keys = {k for k in full if k.startswith(prefix)}
    head_keys = set(full) - backbone_keys
    if encoder is not None:
        p = Path(encoder)
        enc_sd = _validate_tensor_state_dict(_load_raw_state_dict(p), p)
        _overlay_group(
            full, enc_sd, backbone_keys,
            arg_name=encoder_arg, group_desc="encoder-backbone", path=p,
        )
    if head is not None:
        p = Path(head)
        head_sd = _validate_tensor_state_dict(_load_raw_state_dict(p), p)
        _overlay_group(
            full, head_sd, head_keys,
            arg_name=head_arg, group_desc="head", path=p,
        )
    _strict_load_net(net, full)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` UP to the next non-zero multiple of ``multiple``.

    Shared with the encoder's conform helper. Kept here so the model layer
    has no encoder.dinov2 import dependency.
    """
    if value <= 0:
        return int(multiple)
    return int(((value + multiple - 1) // multiple) * multiple)


def _conform_pad_xy(
    x: torch.Tensor,
    *,
    stride: int,
) -> torch.Tensor:
    """Reflect-else-replicate pad XY (bottom/right) to a square multiple of stride.

    Operates on a channel-less 4D ``[B, Z, Y, X]`` tensor. Folds ``(B, Z) → B*Z``
    (adding a singleton channel) so the pad touches only XY, pads ``(0, pad_x, 0,
    pad_y)`` bottom/right on the 4D view, then unfolds. Per-axis mode = ``reflect``
    when the pad amount is ``<`` the corresponding input axis length, else
    ``replicate`` (PyTorch ``reflect`` requires pad < dim). The pad is bottom/right,
    so the un-conform crop (``_restore``) is the trivial ``[..., :Y, :X]`` slice.

    Returns the padded ``[B, Z, target, target]`` tensor.
    """
    B, Z, H, W = x.shape
    target = _ceil_to_multiple(max(H, W), stride)
    pad_y = target - H
    pad_x = target - W
    if pad_y == 0 and pad_x == 0:
        return x

    # Fold (B, Z) → B*Z with a singleton channel for a 4D pad.
    x4 = x.reshape(B * Z, 1, H, W)

    # PER-AXIS mode: ``reflect`` requires the pad amount on each axis < that axis'
    # input length; ``replicate`` has no such constraint.
    needs_replicate_y = pad_y >= H
    needs_replicate_x = pad_x >= W
    if pad_x > 0:
        x4 = torch.nn.functional.pad(
            x4,
            (0, pad_x, 0, 0),
            mode="replicate" if needs_replicate_x else "reflect",
        )
    if pad_y > 0:
        x4 = torch.nn.functional.pad(
            x4,
            (0, 0, 0, pad_y),
            mode="replicate" if needs_replicate_y else "reflect",
        )

    return x4.reshape(B, Z, target, target)


def _conform_resize_xy(
    x: torch.Tensor,
    *,
    stride: int,
) -> torch.Tensor:
    """Bicubic-resize XY (fold-XY-only) of a channel-less 4D ``[B, Z, Y, X]`` tensor.

    Z is never resized. Fold ``(B, Z) → B*Z`` (singleton channel), run 4D bicubic on
    ``[B*Z, 1, Y, X]``, then unfold. Resized to a square
    ``target = ceil(max(Y, X) / stride) * stride``. A no-op when already-conforming.

    Returns the resized ``[B, Z, target, target]`` tensor.
    """
    B, Z, H, W = x.shape
    target = _ceil_to_multiple(max(H, W), stride)
    if H == target and W == target:
        return x
    x4 = x.reshape(B * Z, 1, H, W).to(dtype=torch.float32)
    x4 = torch.nn.functional.interpolate(
        x4,
        size=(target, target),
        mode="bicubic",
        align_corners=False,
    )
    return x4.reshape(B, Z, target, target).to(dtype=x.dtype)


__all__ = ["OmniEM"]

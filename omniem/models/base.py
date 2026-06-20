"""``OmniEM`` — the user-facing model wrapper.

Public surface:

    OmniEM.load(config, weights=None, *, backbone=None, head=None,
                device=None, dtype=None) -> OmniEM
    OmniEM.from_config(config) -> OmniEM                 # random init, test path
    model.predict(x, *, axes, norm=None) -> torch.Tensor # RAW model output
    model.save_weights(path=None, *, backbone=None, head=None) -> Path | (Path, Path)
    model.config, model.device, model.dtype, model.mean, model.std

There is NO bundle, NO meta, NO key rename, NO tag. The model is built from the
user's :class:`ModelConfig` (the recipe) + raw weights file(s); the only check at
load is :meth:`torch.nn.Module.load_state_dict` ``strict=True``.

Internals:

* ``self._net`` is the :class:`OmniEMV1Net` (encoder + STAdapter
  z-fusion + UNETR decoder). The backbone lives at ``self._net.vit``.
* :meth:`predict` returns the raw model output (post the in-model
  ``output_nonlinear``, removed) — the output stage is the
  model method :meth:`OmniEM.apply_output`, gated by ``config.task_type``.
* Normalisation is owned by the MODEL: ``norm=None`` resolves
  to ``config.mean`` / ``config.std`` (the FIXED training norm — the head's own
  training statistics, NOT arch-derived, NOT per-image).
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
from omniem.prepared import Prepared, channel_axis_from_axes

ArrayLike = torch.Tensor | np.ndarray

# Valid axes characters for ``predict``. Same alphabet as :meth:`EMEncoder.forward`.
_AXES_VALID: frozenset[str] = frozenset({"b", "c", "z", "y", "x"})

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

    # ---- apply_input + predict ----------------------------------------------------

    def apply_input(
        self,
        x: ArrayLike,
        *,
        axes: str,
        norm: None | str | Mapping[str, float | Sequence[float]] = None,
        conform: str = "strict",
    ) -> Prepared:
        """Run only the input-transform stage; return a :class:`Prepared` carrier.

        The split-out front half of :meth:`predict`: it owns the wrapper-level
        validation (fp16-CPU, int-dtype, ndarray→tensor), the axes
        canonicalisation, the gray→``in_chans`` channel synthesis, the mean/std
        affine, and the XY conform. The device/dtype move stays in
        :meth:`predict`, so the returned :class:`Prepared` is intentionally
        device-neutral — a split caller can inspect or cache it on CPU before the
        model touches it.

        Args:
            x: Raw grayscale input — a float :class:`torch.Tensor` or
                :class:`numpy.ndarray` (integer dtypes are rejected; the caller
                scales int→float).
            axes: One-character-per-axis string from ``{b,c,z,y,x}`` describing
                ``x`` (same alphabet as :meth:`predict`).
            norm: ``None`` → use ``config.mean``/``config.std``; ``'per-image'`` →
                per-sample z-score (scale-invariant); ``'prenormalized'`` → skip
                the affine; ``{'mean': m, 'std': s}`` → override. Out-of-``[0,1]``
                mean/std or scaled input → warn-only
                :class:`~omniem.errors.OmniEMWarning`.
            conform: ``'strict'`` (default — reject non-square /
                non-stride-multiple XY), ``'pad'`` (bottom/right
                reflect-else-replicate to the next square multiple of the arch's
                XY divisor; cropped back on un-conform), or ``'resize'`` (bicubic
                XY-only interpolate; resized back on un-conform).

        Returns:
            A frozen :class:`~omniem.prepared.Prepared` carrying the canonical
            ``[B, C, Y, X', Z]`` tensor (``C == in_chans``; Z kept for the CNN
            stem / z-fusion) plus the metadata :meth:`predict` needs to invert the
            conform.

        Raises:
            InputContractError: fp16-on-CPU; a non-float / wrong-type input; an
                axes string that does not match ``x``; or a non-conforming XY
                under ``conform='strict'``.
        """
        # fp16 CPU guard — apply_input owns the wrapper
        # validation. Without this guard a caller using the split
        # path would hit a baffling kernel-not-implemented error later.
        if self.dtype == torch.float16 and self.device.type == "cpu":
            raise InputContractError(
                "OmniEM is float16 on CPU — most CPU conv kernels do not support fp16."
            )
        x = self._coerce_input(x)
        cleaned = _parse_axes(axes)
        return self._build_prepared(x, cleaned=cleaned, norm=norm, conform=conform)

    def predict(
        self,
        x: ArrayLike | Prepared,
        *,
        axes: str | None = None,
        norm: None | str | Mapping[str, float | Sequence[float]] = None,
        conform: str = "strict",
    ) -> torch.Tensor:
        """Single-shot model forward — returns RAW model output at caller shape.

        ``x`` may be a raw tensor/ndarray OR a :class:`Prepared`
        returned by :meth:`apply_input`. The type is the signal — no ``prepared=``
        flag. Both forms execute ``compute → un-conform → caller-axes`` and
        return logits at the caller's **original** Y,X — so
        ``predict(apply_input(x)) == predict(x)`` exactly (split == one-shot).

        When ``x`` is a :class:`Prepared`, ``axes`` / ``norm`` / ``conform`` MUST
        be left at their defaults (they were already applied during
        :meth:`apply_input`).

        Args:
            x: Raw input (tensor/ndarray) OR a :class:`Prepared`.
            axes: One-character-per-axis string from ``{b,c,z,y,x}`` (required
                only when ``x`` is raw). Output mirrors ``axes`` with a ``c``
                axis ALWAYS inserted right after ``b`` if present else first:
                ``bzyx``→``bczyx``; ``zyx``→``czyx``; ``yx``→``cyx``.
            norm: One of:
                * ``None`` (default) — use ``config.mean``/``config.std``;
                * ``'per-image'`` — per-sample z-score (scale-invariant);
                * ``'prenormalized'`` — skip the affine;
                * ``{'mean': m, 'std': s}`` — override.
            conform: ``'strict'`` (default — reject non-conforming XY),
                ``'pad'`` (bottom/right reflect-else-replicate; lossless
                round-trip), or ``'resize'`` (bicubic; lossy round-trip).

        Returns:
            ``torch.Tensor`` on ``self.device`` — **pure logits** at the
            caller's ORIGINAL shape. For the canonical task output,
            call :meth:`apply_output` on this tensor.
        """
        if self.dtype == torch.float16 and self.device.type == "cpu":
            raise InputContractError(
                "OmniEM is float16 on CPU — most CPU conv kernels do not support fp16."
            )

        if isinstance(x, Prepared):
            # Prepared-mode: axes/norm/conform are baked in — reject overrides.
            if axes is not None:
                raise InputContractError(
                    "OmniEM.predict: `axes` must be omitted when x is a Prepared "
                    "(axes is baked into Prepared.axes)."
                )
            if norm is not None:
                raise InputContractError(
                    "OmniEM.predict: `norm` must be omitted when x is a Prepared "
                    "(the affine was already applied during apply_input)."
                )
            if conform != "strict":
                raise InputContractError(
                    "OmniEM.predict: `conform` must be omitted when x is a Prepared "
                    "(the conform was already applied during apply_input)."
                )
            self._validate_prepared(x)
            prepared = x
        else:
            if axes is None:
                raise InputContractError(
                    "OmniEM.predict: `axes` is required for raw input (pass a Prepared "
                    "via x= to skip; see apply_input)."
                )
            x = self._coerce_input(x)
            cleaned = _parse_axes(axes)
            prepared = self._build_prepared(x, cleaned=cleaned, norm=norm, conform=conform)

        x5d = prepared.tensor
        # Move + cast onto the model.
        if x5d.device != self.device or x5d.dtype != self.dtype:
            x5d = x5d.to(device=self.device, dtype=self.dtype)

        # Forward — OmniEMV1Net returns [B, C_out, Y, X', Z'] (PURE LOGITS).
        raw = self._net(x5d)

        # Un-conform XY back to the caller's original (orig_yx): the un-conform
        # lives in predict so both raw and Prepared forms return caller-shape.
        raw = self._unconform_xy(raw, prepared=prepared)

        # Reshape to caller axes. Output ALWAYS carries a `c` axis (out_channels)
        # — `c` collapses only in apply_output (image2image squeeze C=1; image2label argmax).
        return self._to_caller_axes(raw, cleaned=prepared.axes, B=prepared.B)

    # ---- output stage ------------------------------------------------------------

    def apply_output(
        self,
        logits: torch.Tensor,
        *,
        axes: str,
        dtype: str = "uint8",
    ) -> torch.Tensor:
        """Model-owned output stage — task_type-gated, derived transform.

        ``axes`` is the
        SAME axes string the caller gave :meth:`predict`; the channel axis is
        derived via :func:`omniem.prepared.channel_axis_from_axes` (``c`` after
        ``b`` if present, else axis 0) — the single source of truth shared with
        :meth:`_to_caller_axes`. Runs only on caller-axes logits (the output of
        :meth:`predict`, one-shot or :class:`Prepared` form).

        Requires :attr:`self.config.task_type` (else :class:`InputContractError`).
        The transform is derived from ``task_type``, never passed:

        * ``image2image`` → ``sigmoid → clamp[0,1] → ·(2ⁿ-1) → round → dtype``
          then **squeeze the C=1 channel** (the single-channel collapse).
        * ``image2label`` → ``argmax`` over the channel axis → class map as
          ``dtype``. The 1-ch threshold branch is gone (image2label is
          ``out_channels>=2`` by construction).

        Both task outputs **collapse the channel axis**.

        Args:
            logits: The raw model output from :meth:`predict`. Shape mirrors
                ``axes`` with a ``c`` axis inserted right after ``b`` if
                present, else first (``yx→cyx``, ``byx→bcyx``, ``zyx→czyx``,
                ``bzyx→bczyx``). ``logits.shape[channel_axis]`` must equal
                ``self.config.out_channels``.
            axes: The caller's axes string (same one passed to :meth:`predict`).
            dtype: ``'uint8'`` (default) or ``'uint16'``. For ``image2label``,
                ``out_channels - 1`` must fit in the chosen dtype.

        Returns:
            A :class:`torch.Tensor` with the channel axis collapsed. Device
            unchanged.

        Raises:
            InputContractError: ``task_type`` is unset; ``dtype`` unknown; the
                logits channel-axis size disagrees with
                ``self.config.out_channels``; or the image2label class count
                overflows ``dtype``.
        """
        from omniem.models import output as _output  # local — avoids import cycle on package init

        task_type = self._config.task_type
        if task_type is None:
            raise InputContractError(
                "model.apply_output requires config.task_type "
                "(\"image2image\" or \"image2label\"); the model has no task_type "
                "so it cannot decide the output transform — postprocess the raw "
                "logits yourself."
            )

        if not isinstance(logits, torch.Tensor):
            raise InputContractError(
                f"apply_output expects a torch.Tensor (got {type(logits).__name__})"
            )
        # predict returns float logits; an integer tensor is a contract violation
        # (image2image would silently round to 0/255 after sigmoid; image2label
        # would silently accept ints "as logits"). Reject loudly.
        if not logits.is_floating_point():
            raise InputContractError(
                f"apply_output expects FLOAT logits (got dtype={logits.dtype}). "
                f"OmniEM.predict returns floats; integer tensors are not valid logits."
            )
        # Resolve dtype name early so unknown values surface with a clear error.
        _output._resolve_dtype(dtype)

        # Derive the channel axis from the caller's `axes` (shared rule).
        # The single source of truth lives in
        # ``channel_axis_from_axes`` so this matches ``_to_caller_axes`` exactly.
        cleaned_axes = _parse_axes(axes)
        ch_axis = channel_axis_from_axes(cleaned_axes)
        # Validate the EXACT rank of caller-axes logits. The
        # rule mirrors what _to_caller_axes constructs: a `c` axis inserted at
        # ch_axis (right after `b` if present, else axis 0), then the SPATIAL
        # axes from the caller's axes string. Input `c` (if any) is ignored —
        # predict drops it before inserting the output channel.
        spatial = sum(1 for a in cleaned_axes if a in ("y", "x", "z"))
        leading_b = 1 if "b" in cleaned_axes else 0
        expected_ndim = leading_b + 1 + spatial  # +1 for the inserted c axis
        if logits.ndim != expected_ndim:
            raise InputContractError(
                f"apply_output (axes={axes!r}): caller-axes logits ndim must be "
                f"{expected_ndim} (leading b={leading_b} + c=1 + spatial={spatial}); "
                f"got shape {tuple(logits.shape)}. apply_output runs only on the "
                f"output of predict — pass the same axes string you gave predict."
            )
        actual_c = int(logits.shape[ch_axis])
        expected_c = int(self._config.out_channels)
        if actual_c != expected_c:
            raise InputContractError(
                f"apply_output (axes={axes!r}): logits channel-axis size "
                f"{actual_c} (dim {ch_axis}) disagrees with "
                f"config.out_channels={expected_c}"
            )

        if task_type == "image2image":
            # config validator ensured out_channels == 1; defensive recheck.
            if expected_c != 1:
                raise InputContractError(
                    f"task_type='image2image' implies out_channels==1; got {expected_c}"
                )
            return _output._apply_image2image(logits, ch_axis=ch_axis, dtype=dtype)

        if task_type == "image2label":
            # Label dtype overflow: highest class id is out_channels - 1; the
            # caller must use a dtype that can hold it.
            full_scale, _ = _output._resolve_dtype(dtype)
            if expected_c - 1 > full_scale:
                raise InputContractError(
                    f"task_type='image2label' with out_channels={expected_c} "
                    f"does not fit in dtype={dtype!r} (max class id "
                    f"{expected_c - 1} > {full_scale}); use 'uint16'."
                )
            return _output._apply_image2label(logits, ch_axis=ch_axis, dtype=dtype)

        # Defensive: Literal already constrains task_type.
        raise InputContractError(f"Unknown task_type={task_type!r}")

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

    def _build_prepared(
        self,
        image: torch.Tensor,
        *,
        cleaned: str,
        norm: None | str | Mapping[str, float | Sequence[float]],
        conform: str,
    ) -> Prepared:
        """Axes fold → conform XY → channel synthesis → normalise → :class:`Prepared`.

        Axes fold → conform XY → channel synthesis → normalise → Prepared.
        The conform step (strict/pad/resize) sits between the axes
        fold and the channel synthesis so the conform math always runs on a
        clean ``[B, 1, Y, X, Z]`` grid.
        """
        if image.ndim != len(cleaned):
            raise InputContractError(
                f"image.ndim ({image.ndim}) does not match axes={cleaned!r} "
                f"(length {len(cleaned)})"
            )

        # Reorder to canonical [B, C, Y, X, Z], adding singletons for missing axes.
        order = ["b", "c", "y", "x", "z"]
        present = set(cleaned)
        perm = [cleaned.index(ax) for ax in order if ax in present]
        x = image.permute(*perm) if perm else image
        new_shape: list[int] = []
        idx = 0
        for ax in order:
            if ax in present:
                new_shape.append(x.shape[idx])
                idx += 1
            else:
                new_shape.append(1)
        x = x.reshape(*new_shape)  # [B, C, Y, X, Z]

        B, C, H, W, Z = x.shape
        orig_yx = (int(H), int(W))

        # Empty-axis guard — H == 0 / W == 0 / B == 0 would
        # pass the strict 0 % stride == 0 check and then make F.pad / interpolate
        # surface raw torch errors. Reject at the wrapper layer with a clear
        # message instead.
        if B <= 0 or H <= 0 or W <= 0:
            raise InputContractError(
                f"OmniEM.apply_input: empty spatial / batch axis (got B={B}, "
                f"Y={H}, X={W}); inputs must have positive size."
            )

        # Z must equal config.img_z exactly (flexible Z is not supported).
        expected_z = self._config.img_z
        if Z != expected_z:
            raise InputContractError(
                f"Z must equal config.img_z={expected_z}; got Z={Z}. "
                f"Flexible/same-padded Z is not supported."
            )

        # Per-image: capture per-sample stats on the PRE-CONFORM input; applied at
        # the norm slot below. The scaled-input [0,1] range warning fires for
        # model/argument norm only.
        per_image = isinstance(norm, str) and norm == _PER_IMAGE
        pi_mean: torch.Tensor | None = None
        pi_std: torch.Tensor | None = None
        if per_image:
            pi_mean, pi_std = _per_image_stats(x.to(torch.float32))
        elif norm != _PRENORMALIZED:
            _warn_tensor_out_of_unit_range(x.to(torch.float32), what="OmniEM input")

        stride = self._stride
        pad_or_scale: dict = {}
        if conform == "strict":
            # XY square + multiple of stride (omniemv1: 112 = lcm(ViT 14, omniem 16)).
            if H != W:
                raise InputContractError(
                    f"XY must be square (got Y={H}, X={W}); EM is in-plane isotropic."
                )
            if H % stride != 0:
                raise InputContractError(
                    f"XY side must be a multiple of stride={stride}; got {H}."
                )
        elif conform == "pad":
            x, pad_y, pad_x, new_h, new_w = _conform_pad_xy(x, stride=stride)
            pad_or_scale = {
                "pad_y": int(pad_y),
                "pad_x": int(pad_x),
                "target": int(new_h),
            }
            H, W = new_h, new_w
        elif conform == "resize":
            x, target = _conform_resize_xy(x, stride=stride)
            pad_or_scale = {"target": int(target)}
            H = W = target
        else:
            raise InputContractError(
                f"conform must be one of 'pad', 'resize', 'strict' (got {conform!r})"
            )

        # Channels — EM is grayscale: C in {1, in_chans}; synthesize to in_chans.
        in_chans = int(self._net.vit.patch_embed.in_chans)
        if C == 1:
            x = x.repeat(1, in_chans, 1, 1, 1)
            C = in_chans
        elif C != in_chans:
            raise InputContractError(
                f"expected C in {{1, {in_chans}}} (EM is grayscale); got C={C}."
            )

        # Float cast for the normalisation arithmetic.
        x = x.to(torch.float32)

        # Normalise once — config.mean/std unless `prenormalized` / override /
        # per-image. Per-image uses the PRE-CONFORM stats (broadcast over B).
        if per_image:
            assert pi_mean is not None and pi_std is not None
            m = pi_mean.to(device=x.device).view(B, 1, 1, 1, 1)
            s = pi_std.to(device=x.device).view(B, 1, 1, 1, 1)
            # Cast back to x.dtype for dtype consistency (x is float32 here, so a no-op;
            # kept symmetric with the encoder path and robust if the cast moves).
            x = ((x - m) / s).to(x.dtype)
        else:
            x = _apply_norm(x, channels=C, config=self._config, norm=norm)

        return Prepared(
            tensor=x,
            axes=cleaned,
            conform=conform,  # type: ignore[arg-type]
            orig_yx=orig_yx,
            pad_or_scale=pad_or_scale,
            B=int(B),
            Z=int(Z),
            stride=int(stride),
        )

    def _validate_prepared(self, prepared: Prepared) -> None:
        """Validate a :class:`Prepared` before consuming it in :meth:`predict`.

        The prepared-mode path takes
        an arbitrary user-constructed (or apply_input-built) ``Prepared`` and
        runs compute / un-conform / remap on it. Reject not just the tensor
        body but also the meta the un-conform/remap trusts (``axes``,
        ``conform``, ``orig_yx``, ``B``, ``Z``), so a malformed Prepared
        surfaces here rather than producing the wrong caller shape.
        """
        # Meta first (cheap) — invalid meta would mislead un-conform/remap.
        if not isinstance(prepared.axes, str) or not prepared.axes:
            raise InputContractError(
                f"Prepared.axes must be a non-empty string (got {prepared.axes!r})"
            )
        # Run the same axes parser predict uses on raw input so axes errors
        # surface with the same diagnostics.
        cleaned_axes = _parse_axes(prepared.axes)
        if prepared.conform not in ("pad", "resize", "strict"):
            raise InputContractError(
                f"Prepared.conform must be 'pad'/'resize'/'strict' "
                f"(got {prepared.conform!r})"
            )
        if not (isinstance(prepared.orig_yx, tuple) and len(prepared.orig_yx) == 2):
            raise InputContractError(
                f"Prepared.orig_yx must be a (Y, X) tuple (got {prepared.orig_yx!r})"
            )
        oy, ox = prepared.orig_yx
        if not (isinstance(oy, int) and isinstance(ox, int) and oy > 0 and ox > 0):
            raise InputContractError(
                f"Prepared.orig_yx must be positive ints (got {prepared.orig_yx!r})"
            )

        t = prepared.tensor
        if not isinstance(t, torch.Tensor):
            raise InputContractError(
                f"Prepared.tensor must be a torch.Tensor (got {type(t).__name__})"
            )
        if not torch.is_floating_point(t):
            raise InputContractError(
                f"Prepared.tensor must be floating-point (got dtype={t.dtype})"
            )
        if t.ndim != 5:
            raise InputContractError(
                f"Model-prepared tensor must be 5D [B, C, Y, X, Z] (got ndim={t.ndim}, "
                f"shape={tuple(t.shape)})"
            )
        in_chans = int(self._net.vit.patch_embed.in_chans)
        if t.shape[1] != in_chans:
            raise InputContractError(
                f"Prepared.tensor channel count {int(t.shape[1])} != in_chans={in_chans} "
                f"(apply_input synthesises the channels — pass raw grayscale to it instead)."
            )
        H, W, Z = int(t.shape[2]), int(t.shape[3]), int(t.shape[4])
        if Z != self._config.img_z:
            raise InputContractError(
                f"Prepared.tensor Z={Z} != config.img_z={self._config.img_z}."
            )
        stride = self._stride
        if H != W or H <= 0 or H % stride != 0:
            raise InputContractError(
                f"Prepared.tensor XY must be square + multiple of stride={stride} "
                f"(got {H}x{W})."
            )
        if prepared.stride != stride:
            raise InputContractError(
                f"Prepared.stride={prepared.stride} disagrees with model stride={stride}. "
                f"(encoder Prepared.stride=14 / ViT patch; model Prepared.stride=112 / "
                f"omniemv1 input divisor.)"
            )
        # B/Z consistency vs the tensor shape so the un-conform
        # / _to_caller_axes that read the meta don't drop or duplicate rows.
        if prepared.B != int(t.shape[0]):
            raise InputContractError(
                f"Prepared.B={prepared.B} != tensor batch dim {int(t.shape[0])}."
            )
        if prepared.Z != int(t.shape[4]):
            raise InputContractError(
                f"Prepared.Z={prepared.Z} != tensor Z dim {int(t.shape[4])}."
            )
        # Omitted-axis singletons. _to_caller_axes drops B if
        # axes lacks 'b' (raw[0]) and squeezes the trailing Z if axes lacks 'z'
        # — but `squeeze(-1)` is a no-op when Z > 1, and dropping raw[0] for
        # B > 1 silently loses data. Reject a Prepared whose tensor has a non-1
        # B / Z that the caller's axes string did not declare.
        if "b" not in cleaned_axes and int(t.shape[0]) != 1:
            raise InputContractError(
                f"Prepared.axes={prepared.axes!r} has no 'b' but tensor batch dim "
                f"is {int(t.shape[0])} > 1 — _to_caller_axes would silently drop "
                f"all but raw[0]. Add 'b' to axes or batch one tile at a time."
            )
        if "z" not in cleaned_axes and int(t.shape[4]) != 1:
            raise InputContractError(
                f"Prepared.axes={prepared.axes!r} has no 'z' but tensor Z dim is "
                f"{int(t.shape[4])} > 1 — _to_caller_axes would silently keep the "
                f"Z axis. Add 'z' to axes."
            )
        # Strict orig_yx must match the tensor (no transform happened); pad/resize
        # orig_yx must NOT exceed the conformed XY.
        if prepared.conform == "strict":
            if (oy, ox) != (H, W):
                raise InputContractError(
                    f"Prepared.orig_yx={prepared.orig_yx} but conform='strict' and the "
                    f"tensor is {H}x{W} (strict is a no-op round-trip)."
                )
        else:
            if oy > H or ox > W:
                raise InputContractError(
                    f"Prepared.orig_yx={prepared.orig_yx} exceeds conformed XY ({H}x{W})."
                )

    def _unconform_xy(
        self,
        raw: torch.Tensor,
        *,
        prepared: Prepared,
    ) -> torch.Tensor:
        """Reverse the conform XY step so logits land at the caller's original (Y, X).

        The un-conform lives in predict so both forms
        (one-shot and split) return caller-shape. ``'strict'`` is a no-op
        round-trip (orig_yx already matches).

        Args:
            raw: ``[B, C_out, Y', X', Z']`` from the net (Y', X' may be the
                conformed side).
            prepared: the :class:`Prepared` whose meta describes how to invert
                the conform.

        Returns:
            ``[B, C_out, orig_Y, orig_X, Z']``.
        """
        conform = prepared.conform
        orig_y, orig_x = prepared.orig_yx
        if conform == "strict":
            return raw
        if conform == "pad":
            # Bottom/right pad → crop is the trivial last-2 slice.
            return raw[..., :orig_y, :orig_x, :]
        if conform == "resize":
            # Bicubic XY-only resize back — fold (B, Z) → B*Z and use 4D interpolate.
            B, C, Hp, Wp, Zd = raw.shape
            x4 = raw.permute(0, 4, 1, 2, 3).reshape(B * Zd, C, Hp, Wp).to(dtype=torch.float32)
            x4 = torch.nn.functional.interpolate(
                x4,
                size=(int(orig_y), int(orig_x)),
                mode="bicubic",
                align_corners=False,
            )
            return (
                x4.reshape(B, Zd, C, orig_y, orig_x)
                .permute(0, 2, 3, 4, 1)
                .to(dtype=raw.dtype)
            )
        raise InputContractError(
            f"_unconform_xy: unknown conform mode {conform!r}"
        )

    def _to_caller_axes(
        self,
        raw: torch.Tensor,
        *,
        cleaned: str,
        B: int,
    ) -> torch.Tensor:
        """Reshape raw ``[B, C_out, Y, X, Z]`` to caller-axes order.

        Output shape MIRRORS the caller's ``axes`` — a leading
        ``b`` in ``axes`` keeps a batch axis (even at B==1); no ``b`` → no batch
        axis. A ``c`` axis is ALWAYS present (the model's ``out_channels``)
        inserted right after ``b`` if present, else leading. Spatial axes
        (``y``/``x``/``z``) follow the caller's order.

        Examples:
            * ``axes='yx'`` → ``[C, Y, X]`` (Z dropped — caller didn't ask for z)
            * ``axes='zyx'`` → ``[C, Z, Y, X]``
            * ``axes='bzyx'`` → ``[B, C, Z, Y, X]``
            * ``axes='byx'`` → ``[B, C, Y, X]``
            * ``axes='cyx'`` → ``[C, Y, X]`` (input c is ignored for output layout)
        """
        has_b = "b" in cleaned
        has_z = "z" in cleaned
        # raw is [B, C_out, Y, X, Z].
        # Drop Z if the caller didn't include it.
        if not has_z:
            raw = raw.squeeze(-1)  # [B, C, Y, X]
        # Drop B if the caller didn't include it (single-shot view).
        if not has_b:
            raw = raw[0]  # [C, ...]
            internal = ["c"] + (["y", "x", "z"] if has_z else ["y", "x"])
            spatial_order = [ax for ax in cleaned if ax in ("y", "x", "z")]
            target = ["c", *spatial_order]
        else:
            internal = ["b", "c"] + (["y", "x", "z"] if has_z else ["y", "x"])
            spatial_order = [ax for ax in cleaned if ax in ("y", "x", "z")]
            target = ["b", "c", *spatial_order]
        return _reorder_axes(raw, internal=internal, target=target)


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
    channels: int,
    config: ModelConfig,
    norm: None | str | Mapping[str, float | Sequence[float]],
) -> torch.Tensor:
    """Apply ``(x − mean) / std`` per the ``norm=`` directive.

    Handles the fixed/override/skip cases. ``'per-image'`` is NOT handled here — it
    needs the pre-conform tensor and is applied by :meth:`_build_prepared` upstream.

    * ``norm is None`` → use ``config.mean`` / ``config.std`` (the model's
      FIXED training normalisation).
    * ``norm == 'prenormalized'`` → skip the affine (caller already did it).
    * ``norm`` is a mapping with EXACT keys ``{'mean', 'std'}`` → override.
    """
    if isinstance(norm, str):
        if norm == _PRENORMALIZED:
            return x
        # 'per-image' is valid at the public API but is handled in _build_prepared
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

    m_t = _to_channel_tensor(eff_mean, channels, device=x.device, name="mean")
    s_t = _to_channel_tensor(eff_std, channels, device=x.device, name="std", positive=True)
    m_t = m_t.view(1, channels, 1, 1, 1)
    s_t = s_t.view(1, channels, 1, 1, 1)
    return (x - m_t) / s_t


def _to_channel_tensor(
    v: float | Sequence[float],
    channels: int,
    *,
    device: torch.device,
    name: str = "mean/std",
    positive: bool = False,
) -> torch.Tensor:
    """Scalar / per-channel mean-or-std → ``[channels]`` tensor, validated."""
    if isinstance(v, bool):
        raise InputContractError(f"{name} must be numeric (got bool)")
    if isinstance(v, (int, float)):
        f = float(v)
        if not math.isfinite(f):
            raise InputContractError(f"{name} must be finite (got {v!r})")
        if positive and f <= 0:
            raise InputContractError(f"{name} must be strictly positive (got {v!r})")
        return torch.full((channels,), f, device=device, dtype=torch.float32)
    if isinstance(v, (str, bytes, bytearray)):
        raise InputContractError(f"{name} must be numeric, not a string (got {v!r})")
    if not isinstance(v, Sequence):
        raise InputContractError(
            f"{name} must be a scalar or numeric per-channel sequence "
            f"(got {type(v).__name__})"
        )
    try:
        vec = [float(t) for t in v]
    except (TypeError, ValueError) as e:
        raise InputContractError(
            f"{name} per-channel sequence must be numeric (got {v!r})"
        ) from e
    if not all(math.isfinite(t) for t in vec):
        raise InputContractError(f"{name} values must be finite (got {v!r})")
    if positive and any(t <= 0 for t in vec):
        raise InputContractError(f"{name} values must be strictly positive (got {v!r})")
    arr = torch.tensor(vec, device=device, dtype=torch.float32)
    if arr.numel() == 1:
        return arr.expand(channels).contiguous()
    if arr.numel() != channels:
        raise InputContractError(
            f"{name} length must be 1 or {channels} (got {arr.numel()})"
        )
    return arr


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
) -> tuple[torch.Tensor, int, int, int, int]:
    """Reflect-else-replicate pad XY (bottom/right) to a square multiple of stride.

    Operates on a 5D ``[B, C, Y, X, Z]`` tensor. Folds ``(B, Z) → B*Z`` so the
    pad touches only XY (a naïve 5D ``F.pad`` would need a fragile padding
    tuple), pads ``(0, pad_x, 0, pad_y)`` bottom/right on the 4D view, then
    unfolds. Per-axis mode = ``reflect`` when the pad amount is ``<`` the
    corresponding input axis length, else ``replicate`` (PyTorch ``reflect``
    requires pad < dim).

    Returns:
        ``(padded_5d, pad_y, pad_x, new_h, new_w)``.
    """
    B, C, H, W, Z = x.shape
    target = _ceil_to_multiple(max(H, W), stride)
    pad_y = target - H
    pad_x = target - W
    if pad_y == 0 and pad_x == 0:
        return x, 0, 0, H, W

    # Fold (B, Z) → B*Z for a 4D pad.
    x4 = x.permute(0, 4, 1, 2, 3).reshape(B * Z, C, H, W)

    # PER-AXIS mode. PyTorch ``reflect`` requires the pad
    # amount on EACH axis to be strictly less than that axis' input length;
    # ``replicate`` has no such constraint. To keep the "good" axis on the more
    # natural ``reflect`` mode even when the other axis needs ``replicate``,
    # apply X first then Y (or vice versa), each with its own mode.
    needs_replicate_y = pad_y >= H
    needs_replicate_x = pad_x >= W
    # X first (last two dims of a 4D tensor are H, W → F.pad's first two pad
    # values control W). Then Y on the result.
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

    x5 = x4.reshape(B, Z, C, target, target).permute(0, 2, 3, 4, 1)
    return x5, int(pad_y), int(pad_x), int(target), int(target)


def _conform_resize_xy(
    x: torch.Tensor,
    *,
    stride: int,
) -> tuple[torch.Tensor, int]:
    """Bicubic-resize XY (fold-XY-only) of a 5D ``[B, C, Y, X, Z]`` tensor.

    A naïve 5D ``F.interpolate`` would rescale Z too — that's the wrong
    operation. Fold ``(B, Z) → B*Z``, run 4D bicubic on the resulting
    ``[B*Z, C, Y, X]``, then unfold. Resized to a square ``target = ceil(max(Y,X)
    / stride) * stride``.

    Returns ``(resized_5d, target)``; when already-conforming, the resize is a
    no-op and ``target`` matches ``Y == X``.
    """
    B, C, H, W, Z = x.shape
    target = _ceil_to_multiple(max(H, W), stride)
    if H == target and W == target:
        return x, int(target)
    x4 = x.permute(0, 4, 1, 2, 3).reshape(B * Z, C, H, W).to(dtype=torch.float32)
    x4 = torch.nn.functional.interpolate(
        x4,
        size=(target, target),
        mode="bicubic",
        align_corners=False,
    )
    x5 = (
        x4.reshape(B, Z, C, target, target)
        .permute(0, 2, 3, 4, 1)
        .to(dtype=x.dtype)
    )
    return x5, int(target)


__all__ = ["OmniEM"]

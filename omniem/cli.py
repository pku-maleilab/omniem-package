"""omniem CLI — ``list-encoders`` / ``list-models`` / ``features`` / ``infer`` /
``split`` / ``merge``.

Surface:

    omniem list-encoders
    omniem list-models
    omniem features -i <img> --arch <NAME> --weights <PATH>
                    [--axes AXES] [--color-to-gray]
                    [--blocks i,j] [--want cls,patch,inner]
                    [--scale dtype|max|S | --unit-range]
                    [--norm model|per-image|prenormalized | --mean M --std S]
                    -o <store> [--force]
    omniem infer    -i <img> -m <config.yaml>
                    (--weights <MERGED.pt> | --backbone <B.pt> --head <H.pt>)
                    [--scale dtype|max|S | --unit-range]
                    [--norm model|per-image|prenormalized | --mean M --std S]
                    [--conform strict|pad|resize] [--output-scale F]
                    [--out-dtype uint8|uint16] [--save-logits]
                    -o <out> [--force]
    omniem split    -m <config.yaml> -i <MERGED.pt> --backbone <B.pt> --head <H.pt>
                    [--force]
    omniem merge    -m <config.yaml> --backbone <B.pt> --head <H.pt> -o <MERGED.pt>
                    [--force]

Input contract = two independent axes (see docs/input-format.md):
  * SCALE (input → float [0,1]): --scale dtype (default, ÷ dtype-max) | max
    (÷ input.max()) | <float>; or --unit-range (already [0,1], skip scaling).
  * NORM (how to normalize): --norm model (default, fixed config/arch mean/std) |
    per-image (per-sample z-score) | prenormalized (skip affine AND scaling);
    or --mean/--std (argument override). per-image / prenormalized skip the scale
    axis. The [0,1] range check is warn-only (OmniEMWarning → stderr).

Pinned observable behavior:

* ``list-encoders`` / ``list-models``: print the **static, owner-frozen catalog**
  of arch names shipped with the package, with a short description line.
* ``features``: an encoder is ``--arch`` + ``--weights`` (a raw ``vit.*``
  checkpoint loaded directly). No config file, no tag.
* ``infer``: single-shot OmniEM model inference. Either ``--weights`` (merged
  state_dict) or ``--backbone`` + ``--head`` (split files). Output is gated by
  ``config.task_type``: no ``task_type`` → writes
  float logits only; ``task_type`` set → writes the derived output via
  :meth:`OmniEM.apply_output` (``image2image`` → uint image, channel collapsed;
  ``image2label`` → int label map). ``--out-dtype`` picks ``uint8`` (default) /
  ``uint16``; ``--save-logits`` additionally writes the float logits beside the
  main store. There is **no ``--output`` flag** — the transform is decided by
  ``task_type``, not chosen at the CLI.
* ``split`` / ``merge``: weight-file utilities wrapping the config-based
  :meth:`OmniEM.load` + :meth:`OmniEM.save_weights` round-trip. ``split`` reads a
  merged ``-i`` file and writes the ``--backbone`` + ``--head`` pair; ``merge`` reads
  the ``--backbone`` + ``--head`` pair and writes a merged ``-o`` file. ``-m`` (config)
  is required: the backbone/head boundary is the net's DERIVED encoder prefix, not an
  assumed ``vit.*``. ``split`` prints the two output paths (backbone then head);
  ``merge`` prints the one merged path.
* Bad usage / unknown command: argparse defaults (stderr, exit ``2``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from omniem import __version__
from omniem.encoders.base import EMEncoder
from omniem.encoders.registry import ARCH_REGISTRY, list_encoders
from omniem.errors import OmniEMError, OmniEMWarning
from omniem.models.base import OmniEM
from omniem.models.registry import MODEL_ARCH_REGISTRY, list_models

# Exit codes used by the CLI (stdout=path; errors -> stderr, exit 2).
#   * 0 — success
#   * 2 — every user-facing error: bad usage (argparse), feature-not-yet-available,
#     and runtime contract violations from ``features`` (missing scale, overwrite
#     refused, unsupported ndim, OmniEMError surface, ...). Uniform "won't proceed"
#     code so wrappers can grep on ``rc != 0``.
_EXIT_OK = 0
_EXIT_ERROR = 2
_EXIT_NOT_YET = 2


# --------------------------------------------------------------------------------------
# Shared input scale/norm CLI surface (two independent axes — see docs/input-format.md)
# --------------------------------------------------------------------------------------


def _scale_arg(value: str) -> str | float:
    """argparse type for ``--scale``: keyword ``dtype``/``max`` OR a positive float."""
    if value in ("dtype", "max"):
        return value
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--scale must be 'dtype', 'max', or a positive number (got {value!r})"
        ) from None
    if not math.isfinite(f) or f <= 0:
        raise argparse.ArgumentTypeError(
            f"--scale must be a positive finite number (got {value!r})"
        )
    return f


def _positive_float_arg(value: str) -> float:
    """argparse type for ``--output-scale``: a strictly-positive, FINITE float.

    Rejects non-numeric strings, ``<= 0``, and non-finite values (``inf``/``nan``).
    Non-finite must be rejected here because the downstream resize multiplies the
    XY size by this factor and ``round(dim * inf)`` raises ``OverflowError`` while
    ``round(dim * nan)`` raises ``ValueError`` (round-2 P2-4). Mirrors the
    ``_scale_arg`` validator's contract.
    """
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--output-scale must be a positive number (got {value!r})"
        ) from None
    if not math.isfinite(f) or f <= 0:
        raise argparse.ArgumentTypeError(
            f"--output-scale must be a positive finite number (got {value!r})"
        )
    return f


def _add_input_scale_norm_args(p: argparse.ArgumentParser) -> None:
    """Add the shared input-side flags (scale axis + norm axis) to a subparser.

    Scale axis (input → float in ``[0,1]``):
      ``--scale {dtype|max|<float>}`` (default ``dtype`` = ÷ dtype-max) and
      ``--unit-range`` (skip scaling; the input is already ``[0,1]``).
    Norm axis (how to normalize):
      ``--norm {model|per-image|prenormalized}`` (default ``model``) OR
      ``--mean M --std S`` (argument-decide; mutually exclusive with ``--norm``).
    """
    g = p.add_argument_group("input scaling + normalization")
    g.add_argument(
        "--scale",
        type=_scale_arg,
        default=None,
        metavar="dtype|max|S",
        help=(
            "Bring the input into [0,1]. 'dtype' (default) = divide by the dtype max "
            "(uint8 255, uint16 65535); 'max' = divide by the input's max; a positive "
            "number divides by it. Mutually exclusive with --unit-range."
        ),
    )
    g.add_argument(
        "--unit-range",
        dest="unit_range",
        action="store_true",
        help="Input is already in [0,1]; skip scaling (a [0,1] range check still warns).",
    )
    g.add_argument(
        "--norm",
        default=None,
        choices=["model", "per-image", "prenormalized"],
        help=(
            "Normalization source. 'model' (default) = the config/arch fixed mean/std; "
            "'per-image' = per-sample z-score (scale-invariant); 'prenormalized' = skip "
            "the affine AND scaling (cast only). Mutually exclusive with --mean/--std."
        ),
    )
    g.add_argument(
        "--mean",
        type=float,
        metavar="M",
        help="Override mean (pair with --std) → argument normalization. Excludes --norm.",
    )
    g.add_argument(
        "--std",
        type=float,
        metavar="S",
        help="Override std (pair with --mean). Must be finite and > 0.",
    )


# --------------------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omniem",
        description=(
            "GUI-free inference and utilities for EM-specific encoders and OmniEM "
            "models. list-encoders prints the static support catalog; features "
            "runs single-shot encoder feature extraction; infer runs single-shot "
            "OmniEM model inference."
        ),
    )
    parser.add_argument("--version", action="version", version=f"omniem {__version__}")

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # ---- list-encoders -----------------------------------------------------------
    p_le = sub.add_parser(
        "list-encoders",
        help="Print the static catalog of supported encoder archs + weights URLs.",
    )
    p_le.set_defaults(func=_cmd_list_encoders)

    # ---- list-models -------------------------------------------------------------
    p_lm = sub.add_parser(
        "list-models",
        help="Print the static catalog of supported model archs (e.g. omniemv1).",
    )
    p_lm.set_defaults(func=_cmd_list_models)

    # ---- features ----------------------------------------------------------------
    p_f = sub.add_parser(
        "features",
        help="Run single-shot encoder feature extraction and write to a store.",
        description=(
            "Single-shot encoder forward pass. An encoder is --arch (a registry "
            "name, e.g. emdinov1) + --weights (a raw vit.* checkpoint). Input scaling "
            "and normalization are two independent axes (--scale / --norm); see the "
            "package docs. Integer inputs default to --scale dtype (÷ dtype max)."
        ),
    )
    p_f.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="IMG",
        help="Input image (.tif or .npy). A grayscale EM image is 2D (Y,X) or 3D "
        "(Z,Y,X). --axes can also declare a stored colour channel (c) or a leading "
        "batch (b); without --axes only 2D=yx / 3D=zyx are inferred.",
    )
    p_f.add_argument(
        "--arch",
        required=True,
        metavar="NAME",
        help="Encoder arch name (see `omniem list-encoders`), e.g. emdinov1.",
    )
    p_f.add_argument(
        "--weights",
        required=True,
        metavar="PATH",
        help="Path to the encoder weights — a raw vit.* checkpoint "
        "(e.g. weights/emdino_v3_best_250703.pth).",
    )
    p_f.add_argument(
        "--axes",
        metavar="AXES",
        help="Declare the input layout, any permutation of {b,c,z,y,x} "
        "(e.g. yx, zyx, cyx, czyx, bcyx). Omit → infer 2D=yx / 3D=zyx.",
    )
    p_f.add_argument(
        "--color-to-gray",
        action="store_true",
        dest="color_to_gray",
        help="Average across the channel axis to get grayscale (valid only when "
        "--axes declares a c axis of size > 1). Default behaviour requires "
        "C ∈ {1, 3} with identical channels; this flag relaxes that.",
    )
    p_f.add_argument(
        "--blocks", metavar="i,j,...", help="Comma-separated block indices for the inner tap."
    )
    p_f.add_argument(
        "--want",
        metavar="FIELDS",
        default="cls",
        help="Comma-separated subset of {cls,patch,inner}. Default: cls. "
        "'inner' requires --blocks.",
    )
    _add_input_scale_norm_args(p_f)
    p_f.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="STORE",
        help="Output store path. Suffix .npz (multi-field) or .npy (single field).",
    )
    p_f.add_argument("--force", action="store_true", help="Overwrite an existing store + sidecar.")
    p_f.set_defaults(func=_cmd_features)

    # ---- infer -------------------------------------------------------------------
    p_i = sub.add_parser(
        "infer",
        help="Run single-shot OmniEM model inference and write to a store.",
        description=(
            "Single-shot OmniEM forward pass. -m is a path to a ModelConfig YAML; "
            "EITHER --weights (a merged state_dict .pt) OR --backbone + --head "
            "(split files, loaded directly). Input scaling (--scale/--unit-range) and "
            "normalization (--norm / --mean/--std) are two independent axes; default "
            "is --scale dtype + --norm model (the config's fixed training mean/std). "
            "--output-scale F resizes the input XY by F before inference (super-res; "
            "XY only, warns for 3D). "
            "Output is gated by config.task_type — no task_type → "
            "writes float logits only; task_type set → writes the derived output "
            "via model.apply_output (--out-dtype uint8|uint16; --save-logits also "
            "writes the float logits to <store>.logits.npy)."
        ),
        # Disable prefix matching so the removed --output is not
        # silently re-bound to --output-path via argparse abbreviation.
        allow_abbrev=False,
    )
    p_i.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="IMG",
        help="Input image (.tif or .npy). 2D (Y,X) or 3D (Z,Y,X).",
    )
    p_i.add_argument(
        "-m", "--model", required=True, metavar="CONFIG", help="Path to a ModelConfig YAML."
    )
    # Mutually exclusive: merged (--weights) OR split (--backbone + --head).
    p_i.add_argument(
        "--weights",
        metavar="MERGED.pt",
        help="Path to a MERGED state_dict .pt (mutually exclusive with --backbone/--head).",
    )
    p_i.add_argument(
        "--backbone",
        metavar="VIT.pt",
        help="Path to the vit.* backbone state_dict (split mode; pair with --head).",
    )
    p_i.add_argument(
        "--head",
        metavar="HEAD.pt",
        help=(
            "Path to the head+adapters state_dict (split mode; pair with --backbone). "
            "Loads split weights_split/head_*.pt files directly."
        ),
    )
    _add_input_scale_norm_args(p_i)
    # --output is REMOVED. The output transform is decided by
    # config.task_type (model.apply_output); --out-dtype + --save-logits are the
    # only sub-knobs. argparse default is None so the runtime can distinguish
    # "user passed --out-dtype" from "unset" and reject the no-task_type case.
    p_i.add_argument(
        "--out-dtype",
        dest="out_dtype",
        default=None,
        choices=["uint8", "uint16"],
        help=(
            "Integer dtype for the canonical OUTPUT (when config.task_type is set). "
            "Default: uint8. Rejected when config.task_type is None (output is logits float)."
        ),
    )
    p_i.add_argument(
        "--save-logits",
        action="store_true",
        dest="save_logits",
        help=(
            "Also write the raw float logits to ``<store>.with_suffix('.logits.npy')``. "
            "Only meaningful when config.task_type is set; rejected otherwise."
        ),
    )
    # --conform picks the input-conform mode for non-conforming XY
    # (default 'strict' = reject).
    p_i.add_argument(
        "--conform",
        default="strict",
        choices=["pad", "resize", "strict"],
        help=(
            "Input-conform mode. 'strict' (default) rejects non-square / "
            "non-stride-multiple XY; 'pad' bottom/right reflect-else-replicate to the "
            "next square multiple of the arch stride (crop on un-conform); 'resize' "
            "bicubic XY-only (lossy)."
        ),
    )
    p_i.add_argument(
        "--output-scale",
        type=_positive_float_arg,
        default=None,
        dest="output_scale",
        metavar="F",
        help=(
            "Resize the input XY by factor F (>0, finite) BEFORE inference; the "
            "shape-preserving model then returns its output at the scaled size "
            "(super-resolution upscales, F>1). XY only — Z is never resized; for 3D "
            "(zyx) inputs this warns (anisotropy / no Z alignment). Orthogonal to "
            "--conform: a non-conforming scaled size still needs --conform pad|resize."
        ),
    )
    p_i.add_argument(
        "-o",
        "--output-path",
        required=True,
        metavar="STORE",
        dest="output_path",
        help="Output store path. Suffix .tif/.npy/.npz.",
    )
    p_i.add_argument("--force", action="store_true", help="Overwrite an existing store + sidecar.")
    p_i.set_defaults(func=_cmd_infer)

    # ---- split: merged -> backbone + head ------------------------------------------
    p_split = sub.add_parser(
        "split",
        help="Split a merged OmniEM weight file into backbone + head files.",
        description=(
            "Split a MERGED state_dict (-i) into the backbone + head pair. -m is a "
            "ModelConfig YAML; it is required because the backbone/head boundary is the "
            "net's DERIVED encoder prefix (not an assumed vit.*). Wraps "
            "OmniEM.load(config, weights=-i).save_weights(backbone=, head=). Prints the "
            "two output paths (backbone then head)."
        ),
        allow_abbrev=False,
    )
    p_split.add_argument(
        "-m", "--model", required=True, metavar="CONFIG", help="Path to a ModelConfig YAML."
    )
    p_split.add_argument(
        "-i",
        "--input",
        required=True,
        dest="input",
        metavar="MERGED.pt",
        help="Path to the MERGED state_dict .pt to split.",
    )
    p_split.add_argument(
        "--backbone",
        required=True,
        metavar="B.pt",
        help="Output path for the backbone (encoder-prefix) state_dict.",
    )
    p_split.add_argument(
        "--head",
        required=True,
        metavar="H.pt",
        help="Output path for the head (non-encoder) state_dict.",
    )
    p_split.add_argument(
        "--force", action="store_true", help="Overwrite existing --backbone / --head files."
    )
    p_split.set_defaults(func=_cmd_split)

    # ---- merge: backbone + head -> merged ------------------------------------------
    p_merge = sub.add_parser(
        "merge",
        help="Merge a backbone + head weight pair into one merged file.",
        description=(
            "Merge the --backbone + --head split pair into one MERGED state_dict (-o). "
            "-m is a ModelConfig YAML; required because the load rebuilds the net to "
            "place the split keys. Wraps OmniEM.load(config, backbone=, head=)."
            "save_weights(path=-o). Prints the merged output path."
        ),
        allow_abbrev=False,
    )
    p_merge.add_argument(
        "-m", "--model", required=True, metavar="CONFIG", help="Path to a ModelConfig YAML."
    )
    p_merge.add_argument(
        "--backbone",
        required=True,
        metavar="B.pt",
        help="Path to the backbone (encoder-prefix) state_dict to merge.",
    )
    p_merge.add_argument(
        "--head",
        required=True,
        metavar="H.pt",
        help="Path to the head (non-encoder) state_dict to merge.",
    )
    p_merge.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output",
        metavar="MERGED.pt",
        help="Output path for the MERGED state_dict.",
    )
    p_merge.add_argument(
        "--force", action="store_true", help="Overwrite an existing -o merged file."
    )
    p_merge.set_defaults(func=_cmd_merge)

    return parser


# --------------------------------------------------------------------------------------
# list-encoders — static catalog
# --------------------------------------------------------------------------------------


def _cmd_list_encoders(ns: argparse.Namespace) -> int:  # noqa: ARG001 — argparse hook
    """Print the static catalog of supported encoder archs.

    Format per arch (one block):

        <arch>
          <description>
          Weights: <url-or-"not yet available">

    Sorted by arch name; "private" archs (``_``-prefixed, e.g. the test-only
    ``_test_tiny``) are hidden from the catalog.
    """
    for name in list_encoders():
        info = ARCH_REGISTRY[name]
        url = info.weights_url or "not yet available"
        print(name)
        if info.description:
            print(f"  {info.description}")
        print(f"  Weights: {url}")
    return _EXIT_OK


def _cmd_list_models(ns: argparse.Namespace) -> int:  # noqa: ARG001 — argparse hook
    """Print the static catalog of supported model archs (parallel to list-encoders)."""
    for name in list_models():
        info = MODEL_ARCH_REGISTRY[name]
        print(name)
        if info.description:
            print(f"  {info.description}")
    return _EXIT_OK


# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------


def _cmd_features(ns: argparse.Namespace) -> int:
    """The single-shot ``features`` command.

    Exits non-zero on any user-facing error (writes a friendly message to stderr).
    """
    try:
        return _run_features(ns)
    except OmniEMError as e:
        print(f"omniem features: {e}", file=sys.stderr)
        return _EXIT_ERROR
    except FileNotFoundError as e:
        print(f"omniem features: file not found: {e}", file=sys.stderr)
        return _EXIT_ERROR


def _sidecar_path_for(out_path: Path) -> Path:
    """Compute the JSON sidecar path for an output store.

    Always ``<store>.json`` — i.e. the sidecar carries the full store suffix so
    ``features.npz`` → ``features.npz.json`` (avoids collisions when two stores
    share the same stem with different suffixes).
    """
    return out_path.with_suffix(out_path.suffix + ".json")


# --------------------------------------------------------------------------------------
# features: selector + norm parsing (cheap CLI-side validation, runs before load)
# --------------------------------------------------------------------------------------

_WANT_FIELDS: frozenset[str] = frozenset({"cls", "patch", "inner"})


def _parse_want(raw: str | None) -> tuple[str, ...]:
    """Parse + validate ``--want``. Rejects empty, unknown, and duplicate fields.

    Duplicate selectors are rejected (no silently-ignored repeats).
    """
    want_raw = raw if raw is not None else ""
    want = tuple(s.strip() for s in want_raw.split(",") if s.strip())
    if not want:
        raise OmniEMError(
            f"--want must list at least one of {{cls,patch,inner}} (got {want_raw!r})."
        )
    seen: set[str] = set()
    for w in want:
        if w not in _WANT_FIELDS:
            raise OmniEMError(f"--want got unknown field {w!r}; allowed: cls, patch, inner.")
        if w in seen:
            raise OmniEMError(f"--want has duplicate field {w!r}.")
        seen.add(w)
    return want


def _parse_blocks(raw: str | None) -> tuple[int, ...] | None:
    """Parse ``--blocks`` into a tuple of ints (``None`` when not given)."""
    if not raw:
        return None
    try:
        return tuple(int(s.strip()) for s in raw.split(",") if s.strip())
    except ValueError as e:
        raise OmniEMError(
            f"--blocks must be comma-separated integer block ids; got {raw!r} ({e})"
        ) from e


def _check_inner_blocks(want: tuple[str, ...], blocks: tuple[int, ...] | None) -> None:
    """Enforce the symmetric inner ⇔ blocks rule.

    ``inner`` is returned ONLY when it is in ``--want`` AND ``--blocks`` is given.
    Either one without the other is an error (no silent ignore).
    """
    has_inner = "inner" in want
    has_blocks = bool(blocks)
    if has_inner and not has_blocks:
        raise OmniEMError("--want inner requires --blocks i,j,...")
    if has_blocks and not has_inner:
        raise OmniEMError(
            "--blocks was given but --want does not include 'inner'; add 'inner' to "
            "--want (e.g. --want cls,inner) or drop --blocks."
        )


def _resolve_cli_norm(
    ns: argparse.Namespace,
) -> tuple[None | str | dict[str, float], str]:
    """Map ``--norm`` / ``--mean`` / ``--std`` to the core ``norm=`` value + kind tag.

    Returns ``(norm_value, kind)``:
    * ``--norm model`` / unset → ``(None, 'model')`` (config/arch fixed mean/std);
    * ``--norm per-image`` → ``('per-image', 'per-image')`` (per-sample z-score);
    * ``--norm prenormalized`` → ``('prenormalized', 'prenormalized')`` (skip affine + scale);
    * ``--mean M --std S`` → ``({'mean':M,'std':S}, 'argument')`` (mutually exclusive
      with ``--norm``; both required together; std finite + > 0).
    """
    has_mean = ns.mean is not None
    has_std = ns.std is not None
    explicit_norm = ns.norm is not None
    if has_mean != has_std:
        raise OmniEMError("--mean and --std must be given together.")
    if explicit_norm and (has_mean or has_std):
        raise OmniEMError("--norm is mutually exclusive with --mean/--std (use one).")
    if has_mean:
        m, s = float(ns.mean), float(ns.std)
        if not (math.isfinite(m) and math.isfinite(s)):
            raise OmniEMError("--mean and --std must be finite.")
        if s <= 0:
            raise OmniEMError("--std must be strictly positive (it divides the input).")
        return {"mean": m, "std": s}, "argument"
    src = ns.norm or "model"
    if src == "model":
        return None, "model"
    if src == "per-image":
        return "per-image", "per-image"
    return "prenormalized", "prenormalized"


def _resolve_scale(
    img_np: np.ndarray,
    *,
    source_dtype: np.dtype,
    ns: argparse.Namespace,
    norm_kind: str,
    warn_prefix: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve the scale axis → ``(float32_array, scale_record)``.

    The ``[0,1]`` range warning is the CORE's job (single source); this emits only
    CLI **flag-coupling** notices to stderr. ``per-image``/``prenormalized`` skip
    scaling (cast only). ``--unit-range`` is mutually exclusive with ``--scale``.
    """
    norm_skips = norm_kind in ("per-image", "prenormalized")
    explicit_scale = ns.scale is not None
    unit_range = bool(ns.unit_range)
    if unit_range and explicit_scale:
        raise OmniEMError(
            "--unit-range is mutually exclusive with --scale (both set the scale axis)."
        )
    # The user's EXPLICIT scale request (None when nothing was passed → the dtype
    # default applies). ``--unit-range`` gets its own token so it is recorded even when
    # the effective kind is also 'skip' — an explicit unit-range must round-trip the
    # sidecar like an explicit --scale.
    if unit_range:
        requested: str | None = "unit-range"
    elif explicit_scale:
        requested = ns.scale if isinstance(ns.scale, str) else "specific"
    else:
        requested = None

    def _rec(kind: str, divisor: float | None, input_max: float | None) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "kind": kind,
            "divisor": divisor,
            "source_dtype": str(source_dtype),
            "input_max": input_max,
        }
        # Record the explicit request only when it differs from the effective kind
        # (an override happened); a bare default never writes a spurious request.
        if requested is not None and requested != kind:
            rec["requested"] = requested
        return rec

    if norm_skips:
        if explicit_scale or unit_range:
            print(
                f"{warn_prefix}: --scale/--unit-range ignored under --norm {norm_kind} "
                f"(it skips scaling).",
                file=sys.stderr,
            )
        return img_np.astype(np.float32), _rec("skip", None, None)

    if unit_range:
        return img_np.astype(np.float32), _rec("skip", None, None)
    if explicit_scale and not isinstance(ns.scale, str):
        s = float(ns.scale)
        return img_np.astype(np.float32) / s, _rec("specific", s, None)
    if ns.scale == "max":
        m = float(img_np.max()) if img_np.size else 0.0
        if m <= 0:
            print(
                f"{warn_prefix}: --scale max: input max is {m} (<=0); skipping divide.",
                file=sys.stderr,
            )
            return img_np.astype(np.float32), _rec("max", None, m)
        return img_np.astype(np.float32) / m, _rec("max", m, m)
    # default (no flag) or explicit --scale dtype → divide by the dtype max
    if np.issubdtype(source_dtype, np.integer):
        div = float(np.iinfo(source_dtype).max)
        return img_np.astype(np.float32) / div, _rec("dtype", div, None)
    # bool or float → already [0,1] / caller-owned → cast only.
    return img_np.astype(np.float32), _rec("dtype", None, None)


def _norm_record(
    kind: str,
    norm: None | str | dict[str, float],
    *,
    default_mean: float,
    default_std: float,
) -> dict[str, Any]:
    """Build the sidecar ``norm`` object.

    ``kind`` ∈ ``{model, per-image, prenormalized, argument}``. ``model`` records the
    effective default mean/std (config for the model, arch for the encoder).
    """
    if kind == "model":
        return {"kind": "model", "mean": default_mean, "std": default_std, "per_image_stats": False}
    if kind == "per-image":
        return {"kind": "per-image", "mean": None, "std": None, "per_image_stats": True}
    if kind == "prenormalized":
        return {"kind": "prenormalized", "mean": None, "std": None, "per_image_stats": False}
    assert isinstance(norm, dict)
    return {"kind": "argument", "mean": norm["mean"], "std": norm["std"], "per_image_stats": False}


def _run_features(ns: argparse.Namespace) -> int:
    out_path = Path(ns.output)
    sidecar_path = _sidecar_path_for(out_path)
    # --force / overwrite guard (B-r2.10).
    if not ns.force:
        if out_path.exists():
            print(
                f"omniem features: {out_path} already exists; pass --force to overwrite.",
                file=sys.stderr,
            )
            return _EXIT_ERROR
        if sidecar_path.exists():
            print(
                f"omniem features: sidecar {sidecar_path} already exists; pass --force.",
                file=sys.stderr,
            )
            return _EXIT_ERROR

    # ---- selectors + norm (CHEAP — parse BEFORE the expensive load so a typo
    #      fails fast and we never pay for the weights load on bad input;
    #      bad input) -----------------------------------------------------------
    want = _parse_want(ns.want)  # rejects empty / unknown / duplicates (finding #5)
    blocks = _parse_blocks(ns.blocks)
    _check_inner_blocks(want, blocks)  # design (B): inner <-> blocks, both or neither
    norm, norm_kind = _resolve_cli_norm(ns)  # --norm / --mean+--std → norm value + kind

    # ---- load encoder ----------------------------------------------------------
    # An encoder is (arch name, weights path). Both argparse-required.
    enc = EMEncoder.load(ns.arch, Path(ns.weights))

    # ---- load image ------------------------------------------------------------
    img_np = _read_image(Path(ns.input))

    # ---- axes resolution (source axes — the layout of the on-disk array) ------
    # Two modes:
    #   * --axes given → require ndim == len(axes); accept any permutation of
    #     {b?, c?, z?, y, x}. ``y`` and ``x`` are required.
    #   * --axes omitted → infer 2D=yx / 3D=zyx (today's behaviour). 3D with a
    #     size-3 axis is ambiguous (could be RGB-stored grayscale) → warn to
    #     stderr but proceed with the zyx inference (exit 0, stdout=path).
    if ns.axes is not None:
        try:
            source_axes = _resolve_explicit_axes(ns.axes, img_np.shape)
        except OmniEMError as e:
            print(f"omniem features: {e}", file=sys.stderr)
            return _EXIT_ERROR
    else:
        if img_np.ndim == 2:
            source_axes = "yx"
        elif img_np.ndim == 3:
            source_axes = "zyx"
            if 3 in img_np.shape:
                # Warn-only (stdout stays clean: just the output path; exit 0).
                print(
                    f"omniem features: input has shape {img_np.shape} with a size-3 axis "
                    f"and no --axes given; inferring 'zyx'. If this is RGB-stored "
                    f"grayscale, pass --axes (e.g. cyx, czyx) to declare the layout.",
                    file=sys.stderr,
                )
        else:
            print(
                f"omniem features: input ndim {img_np.ndim} not supported without --axes "
                f"(expected 2D=yx or 3D=zyx). Pass --axes to declare the layout.",
                file=sys.stderr,
            )
            return _EXIT_ERROR

    # ---- channel reduction (source_axes → gray0 forward_axes) ------------------
    # We reduce on the c axis BEFORE the int→float scale so the scale operates on
    # one channel of data. gray3-pick is exact-equal across channels; --color-to-gray
    # averages (promoting to float internally). The int/float scale decision below
    # must look at the **source** dtype, not the post-reduction dtype — otherwise
    # the mean path's float32 promotion would silently skip --scale on integer
    # inputs.
    source_dtype = img_np.dtype
    try:
        img_np, forward_axes, channel_reduction = _reduce_to_gray0(
            img_np,
            source_axes=source_axes,
            color_to_gray=bool(ns.color_to_gray),
        )
    except OmniEMError as e:
        print(f"omniem features: {e}", file=sys.stderr)
        return _EXIT_ERROR

    # ---- scale axis: input -> float [0,1] (after channel reduction) -----------
    # input-max divides by the reduced gray data's max; dtype-default by the
    # SOURCE dtype max; per-image / prenormalized skip scaling (cast only). The
    # [0,1] range warning is the encoder's job (single source).
    img_np, scale_record = _resolve_scale(
        img_np,
        source_dtype=source_dtype,
        ns=ns,
        norm_kind=norm_kind,
        warn_prefix="omniem features",
    )

    import torch  # local import keeps a future bare-minimum env happy

    image = torch.from_numpy(np.ascontiguousarray(img_np))

    # ---- forward ---------------------------------------------------------------
    fwd_kwargs: dict[str, Any] = {
        "axes": forward_axes,
        "return_cls": "cls" in want,
        "return_patch": "patch" in want,
        "norm": norm,
    }
    # Design (B): inner is fed ONLY when 'inner' is in --want (guaranteed paired
    # with --blocks by _check_inner_blocks). --blocks alone never writes inner.
    if "inner" in want:
        fwd_kwargs["return_blocks"] = list(blocks)
    out = enc(image, **fwd_kwargs)

    # ---- write -----------------------------------------------------------------
    norm_record = _norm_record(norm_kind, norm, default_mean=enc.mean, default_std=enc.std)
    _write_store(out, out_path)
    _write_sidecar(
        sidecar_path,
        enc=enc,
        input_path=Path(ns.input),
        weights_path=Path(ns.weights),
        source_axes=source_axes,
        forward_axes=forward_axes,
        channel_reduction=channel_reduction,
        want=want,
        blocks=list(blocks) if blocks is not None else None,
        scale_record=scale_record,
        norm_record=norm_record,
        color_to_gray=bool(ns.color_to_gray),
        store_path=out_path,
    )
    print(str(out_path))
    return _EXIT_OK


# --------------------------------------------------------------------------------------
# explicit axes + gray-reduction helpers (the "front door" of the CLI)
# --------------------------------------------------------------------------------------


_CLI_AXES_VALID: frozenset[str] = frozenset({"b", "c", "z", "y", "x"})


def _resolve_explicit_axes(axes: str, shape: tuple[int, ...]) -> str:
    """Validate an explicit ``--axes`` string against the image shape.

    Rules:
      * whitespace stripped; every character must be in ``{b, c, z, y, x}``;
      * no duplicates;
      * if ``b`` is present, it must be the first character (matches the encoder's
        canonical leading-batch axis);
      * both ``y`` and ``x`` are required;
      * ``len(axes) == ndim``.
    """
    if not isinstance(axes, str) or not axes:
        raise OmniEMError(f"--axes must be a non-empty string (got {axes!r}).")
    cleaned = "".join(axes.split())
    if not cleaned:
        raise OmniEMError(f"--axes must contain at least one axis (got {axes!r}).")
    seen: set[str] = set()
    for ax in cleaned:
        if ax not in _CLI_AXES_VALID:
            raise OmniEMError(
                f"--axes contains unknown axis {ax!r} (allowed: b, c, z, y, x; got {axes!r})."
            )
        if ax in seen:
            raise OmniEMError(f"--axes has duplicate axis {ax!r} (got {axes!r}).")
        seen.add(ax)
    if "b" in cleaned and cleaned[0] != "b":
        raise OmniEMError(f"--axes: 'b' must be the leading axis when present (got {axes!r}).")
    if "y" not in cleaned or "x" not in cleaned:
        raise OmniEMError(f"--axes must include both 'y' and 'x' (got {axes!r}).")
    if len(cleaned) != len(shape):
        raise OmniEMError(
            f"image ndim ({len(shape)}) does not match --axes {axes!r} (length {len(cleaned)})."
        )
    return cleaned


def _reduce_to_gray0(
    img_np: np.ndarray,
    *,
    source_axes: str,
    color_to_gray: bool,
) -> tuple[np.ndarray, str, str]:
    """Reduce a channel-bearing array to gray0 (channel-less).

    Args:
        img_np: The on-disk array, in ``source_axes`` order.
        source_axes: A cleaned axes string already validated by
            :func:`_resolve_explicit_axes` (or inferred 2D/3D).
        color_to_gray: When ``True``, average across the channel axis to produce
            a grayscale image (valid only when ``c`` is present and size > 1).

    Returns:
        ``(reduced_array, forward_axes, channel_reduction)`` where
        ``forward_axes`` is ``source_axes`` with the ``c`` axis dropped (or the
        same string when no ``c`` was present), and ``channel_reduction`` is one
        of ``{none, squeeze, pick, mean}``.

    Reduction rules:
        * No ``c`` in axes:
            - ``--color-to-gray`` was passed → error (no channel axis to reduce);
            - otherwise pass through (``none``).
        * ``C == 1`` → squeeze the singleton channel axis (``squeeze``).
        * ``C == 3`` (without ``--color-to-gray``):
            - all three channels exactly equal → pick channel 0 (``pick``);
            - channels differ → error (point at ``--color-to-gray`` + the doc).
        * ``C ∉ {1, 3}`` without ``--color-to-gray`` → error.
        * ``--color-to-gray`` with ``C > 1`` (any channel count, incl. 4 / RGBA)
          → mean across ``c`` (promoted to float; ``mean``).
    """
    if "c" not in source_axes:
        if color_to_gray:
            raise OmniEMError(
                "--color-to-gray is only valid when --axes declares a 'c' axis of "
                f"size > 1 (got source_axes={source_axes!r} with no 'c' axis)."
            )
        return img_np, source_axes, "none"

    c_pos = source_axes.index("c")
    C = int(img_np.shape[c_pos])
    forward_axes = source_axes.replace("c", "")

    if color_to_gray:
        if C <= 1:
            raise OmniEMError(f"--color-to-gray requires a 'c' axis of size > 1 (got C={C}).")
        # Promote to float and take the mean across the channel axis.
        promoted = img_np.astype(np.float32, copy=False)
        reduced = promoted.mean(axis=c_pos)
        return reduced, forward_axes, "mean"

    if C == 1:
        return np.squeeze(img_np, axis=c_pos), forward_axes, "squeeze"

    if C == 3:
        # gray3-pick: every channel must be *exactly* equal (bit-identical) — the
        # CLI does not silently average random RGB data. Use np.array_equal so
        # int/float comparison is exact (no rounding).
        ch0 = np.take(img_np, indices=0, axis=c_pos)
        for k in range(1, 3):
            if not np.array_equal(ch0, np.take(img_np, indices=k, axis=c_pos)):
                raise OmniEMError(
                    "gray3 input has differing channel values; pass --color-to-gray "
                    "to average them (see docs/input-format.md)."
                )
        return ch0, forward_axes, "pick"

    raise OmniEMError(
        f"channel axis has size {C}; without --color-to-gray, C must be 1 or 3 "
        f"(EM is grayscale; see docs/input-format.md)."
    )


def _read_image(path: Path) -> np.ndarray:
    """Read a 2D/3D image from .tif or .npy."""
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(str(path))
    if path.suffix.lower() == ".npy":
        return np.load(str(path))
    raise OmniEMError(f"Unsupported input extension {path.suffix!r}; use .tif or .npy")


def _write_store(out: dict[str, Any], out_path: Path) -> None:
    """Persist the encoder's output dict to ``out_path``.

    Flattens ``inner: {i: Tensor}`` into top-level ``inner_<i>`` arrays so the file
    is dict-keyed by string (npz constraint).
    """
    flat: dict[str, np.ndarray] = {}
    for k, v in out.items():
        if k == "inner":
            for i, t in v.items():
                flat[f"inner_{i}"] = t.detach().cpu().float().numpy()
        else:
            flat[k] = v.detach().cpu().float().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".npy":
        if len(flat) != 1:
            raise OmniEMError(f".npy store supports exactly one field (got {sorted(flat.keys())})")
        np.save(out_path, next(iter(flat.values())))
    elif out_path.suffix == ".npz":
        np.savez(out_path, **flat)
    else:
        raise OmniEMError(f"Unsupported output extension {out_path.suffix!r}; use .npz or .npy")


def _write_sidecar(
    sidecar_path: Path,
    *,
    enc: EMEncoder,
    input_path: Path,
    weights_path: Path,
    source_axes: str,
    forward_axes: str,
    channel_reduction: str,
    want: tuple[str, ...],
    blocks: list[int] | None,
    scale_record: dict[str, Any],
    norm_record: dict[str, Any],
    color_to_gray: bool,
    store_path: Path,
) -> None:
    """Write a small JSON sidecar describing the run.

    Field naming:
        * ``arch`` + ``weights`` identify the encoder (an encoder = arch + weights;
          there is no tag).
        * ``source_axes`` is the layout of the on-disk array; ``forward_axes`` is
          what the encoder actually saw after channel reduction (always c-free).
        * ``channel_reduction`` ∈ ``{none, squeeze, pick, mean}``.
        * ``scale`` records the scale axis: ``{"kind": dtype|max|specific|skip,
          "divisor": <n|null>, "source_dtype": str, "input_max": <n|null>,
          ["requested": <kind>]}`` (``requested`` only when norm coupling overrode it).
        * ``norm`` records the normalisation: ``{"kind": model|per-image|argument|
          prenormalized, "mean": <n|null>, "std": <n|null>, "per_image_stats": bool}``.
        * ``encoder_channel_contract`` advertises the gray0 contract.
        * ``want`` mirrors the CLI ``--want`` selector.
    """
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "store": store_path.name,
        "input": str(input_path),
        "source_axes": source_axes,
        "forward_axes": forward_axes,
        "channel_reduction": channel_reduction,
        "color_to_gray": bool(color_to_gray),
        "encoder_channel_contract": "gray0",
        "arch": enc.arch,
        "weights": str(weights_path),
        "embed_dim": enc.embed_dim,
        "want": list(want),
        "blocks": blocks,
        "scale": scale_record,
        "norm": norm_record,
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# --------------------------------------------------------------------------------------
# infer
# --------------------------------------------------------------------------------------


def _cmd_infer(ns: argparse.Namespace) -> int:
    """The single-shot ``infer`` command."""
    try:
        return _run_infer(ns)
    except OmniEMError as e:
        print(f"omniem infer: {e}", file=sys.stderr)
        return _EXIT_ERROR
    except FileNotFoundError as e:
        print(f"omniem infer: file not found: {e}", file=sys.stderr)
        return _EXIT_ERROR


def _run_infer(ns: argparse.Namespace) -> int:
    out_path = Path(ns.output_path)
    sidecar_path = _sidecar_path_for(out_path)
    if not ns.force:
        if out_path.exists():
            print(
                f"omniem infer: {out_path} already exists; pass --force to overwrite.",
                file=sys.stderr,
            )
            return _EXIT_ERROR
        if sidecar_path.exists():
            print(
                f"omniem infer: sidecar {sidecar_path} already exists; pass --force.",
                file=sys.stderr,
            )
            return _EXIT_ERROR

    norm, norm_kind = _resolve_cli_norm(ns)  # --norm / --mean+--std → norm value + kind

    # ---- load model ------------------------------------------------------------
    # Mutually exclusive load modes: merged (--weights) OR split (--backbone + --head).
    merged_given = ns.weights is not None
    split_given = ns.backbone is not None or ns.head is not None
    if merged_given and split_given:
        raise OmniEMError(
            "--weights is mutually exclusive with --backbone/--head."
        )
    if not merged_given and not split_given:
        raise OmniEMError(
            "--weights (merged) OR --backbone + --head (split) is required."
        )
    if split_given and (ns.backbone is None or ns.head is None):
        raise OmniEMError("split mode requires BOTH --backbone and --head.")
    load_kwargs: dict[str, Any] = {}
    if merged_given:
        load_kwargs["weights"] = Path(ns.weights)
    else:
        load_kwargs["backbone"] = Path(ns.backbone)
        load_kwargs["head"] = Path(ns.head)
    model = OmniEM.load(Path(ns.model), **load_kwargs)

    # ---- load image ------------------------------------------------------------
    img_np = _read_image(Path(ns.input))
    if img_np.ndim == 2:
        axes = "yx"
    elif img_np.ndim == 3:
        axes = "zyx"
    else:
        print(
            f"omniem infer: input ndim {img_np.ndim} not supported (expected 2D or 3D).",
            file=sys.stderr,
        )
        return _EXIT_ERROR

    # ---- scale axis: input -> float [0,1] -------------------------------------
    source_dtype = img_np.dtype
    img_np, scale_record = _resolve_scale(
        img_np,
        source_dtype=source_dtype,
        ns=ns,
        norm_kind=norm_kind,
        warn_prefix="omniem infer",
    )

    import torch  # local import

    image = torch.from_numpy(np.ascontiguousarray(img_np))

    # ---- output policy gate (validate cheaply BEFORE any resize / forward) ----
    # task_type unset -> logits-float only; --dtype/--save-logits illegal.
    # task_type set   -> derived transform via model.apply_output(dtype=); the
    #                    CLI calls ONLY model.apply_output (drift guard).
    # These checks are cheap and run BEFORE the --output-scale resize + forward so
    # a bad flag combo / existing logits sibling fails fast — a large --output-scale
    # must not allocate / OOM ahead of a clean exit-2 error. The forward below uses
    # the SPLIT path (apply_input → predict → apply_output) so the sidecar can
    # record `conform` mode + original/conformed XY from the Prepared meta.
    conform = ns.conform
    task_type = model.config.task_type
    dtype_given = ns.out_dtype is not None
    save_logits = bool(ns.save_logits)
    if task_type is None:
        if dtype_given or save_logits:
            print(
                "omniem infer: --out-dtype / --save-logits require a `task_type` in "
                "the config (the model has no opinion on the output transform; "
                "the only output is the float logits).",
                file=sys.stderr,
            )
            return _EXIT_ERROR
        applied_dtype: str | None = None
        logits_path: Path | None = None
        derived_transform: str | None = None
    else:
        # Resolve --out-dtype default only AFTER confirming task_type.
        applied_dtype = ns.out_dtype or "uint8"
        derived_transform = "image" if task_type == "image2image" else "labels"
        # Check the --save-logits sibling existence BEFORE the (expensive) forward
        # pass so the user doesn't wait through inference only to learn the path
        # already exists.
        if save_logits:
            logits_path = out_path.with_suffix(".logits.npy")
            if not ns.force and logits_path.exists():
                print(
                    f"omniem infer: logits sibling {logits_path} already exists; "
                    f"pass --force to overwrite.",
                    file=sys.stderr,
                )
                return _EXIT_ERROR
        else:
            logits_path = None

    # ---- output-size control: --output-scale (input XY pre-resize) -----------
    # The model is shape-preserving (output XY == input XY), so a larger output
    # (e.g. super-resolution) is produced by resizing the INPUT up first; the
    # output then lands at the scaled size with no output-side resize. The resize
    # runs on float [0,1] data (after --scale) with bicubic — the same
    # interpolation as the in-pipeline conform='resize' path. Orthogonal to --conform:
    # a non-conforming scaled size is still rejected/handled by apply_input(conform).
    # Placed AFTER the cheap output-policy validation above so a huge factor cannot
    # allocate / OOM ahead of a clean flag/overwrite error.
    output_scale_record: dict[str, Any] | None = None
    if ns.output_scale is not None:
        import torch.nn.functional as F  # local import

        factor = float(ns.output_scale)
        # axes is 'yx' (Y,X) or 'zyx' (Z,Y,X); XY are always the last two dims.
        in_y, in_x = int(image.shape[-2]), int(image.shape[-1])
        new_y, new_x = round(in_y * factor), round(in_x * factor)
        if new_y < 1 or new_x < 1:
            raise OmniEMError(
                f"--output-scale {factor} shrinks XY ({in_y}x{in_x}) below 1 pixel "
                f"({new_y}x{new_x}); choose a larger factor."
            )
        if axes == "zyx":
            # XY-only resize on an anisotropic volume distorts in-plane spatial
            # information relative to Z, and no Z alignment / resampling is applied
            # (Z is left untouched). The "anisotropy" / "Z alignment" substrings are
            # stable hooks the 3D test asserts on.
            warnings.warn(
                "--output-scale resizes XY only and leaves Z untouched; on a 3D "
                "volume this changes in-plane spatial information relative to Z "
                "(anisotropy) with no Z alignment / resampling applied, and may give "
                "worse results. Not recommended for 3D inputs.",
                OmniEMWarning,
                stacklevel=2,
            )
        # bicubic XY resize. 2D [Y,X] -> add fake batch+channel; 3D [Z,Y,X] -> Z is
        # the batch dim (untouched), add a fake channel for per-slice XY resize.
        if axes == "yx":
            resized = F.interpolate(
                image[None, None], size=(new_y, new_x), mode="bicubic", align_corners=False
            )[0, 0]
        else:  # zyx
            resized = F.interpolate(
                image[:, None], size=(new_y, new_x), mode="bicubic", align_corners=False
            )[:, 0]
        image = resized.contiguous()
        output_scale_record = {
            "factor": factor,
            "input_yx": [in_y, in_x],
            "scaled_yx": [new_y, new_x],
        }

    # ---- forward: single split path (apply_input -> predict -> apply_output) --
    # Uses the SPLIT path so the sidecar can record `conform` mode + original XY +
    # conformed XY from the Prepared meta. CLI passes axes='yx' / 'zyx' (no `b`),
    # so `raw` is unbatched and apply_output derives ch_axis=0 from the same `axes`.
    prepared = model.apply_input(image, axes=axes, norm=norm, conform=conform)
    raw = model.predict(prepared)
    if task_type is None:
        out_tensor = raw
    else:
        out_tensor = model.apply_output(raw, axes=axes, dtype=applied_dtype)

    # ---- write ----------------------------------------------------------------
    _write_predict_store(out_tensor, out_path, derived_transform=derived_transform)
    if logits_path is not None:
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(logits_path, raw.detach().cpu().numpy())
    # Report the conform mode + original/conformed XY
    # from the Prepared tensor directly. Reading the prepared tensor's XY shape
    # avoids re-encoding conform knowledge in the CLI (a future conform mode
    # that produces a non-square conformed shape would not be silently
    # misreported here).
    orig_yx = list(prepared.orig_yx)
    # The model's canonical layout is [B, C, Y, X, Z]; read shape[2:4] directly.
    conformed_yx = [int(prepared.tensor.shape[2]), int(prepared.tensor.shape[3])]
    _write_infer_sidecar(
        sidecar_path,
        model=model,
        input_path=Path(ns.input),
        axes=axes,
        scale_record=scale_record,
        norm_record=_norm_record(norm_kind, norm, default_mean=model.mean, default_std=model.std),
        load_mode="merged" if merged_given else "split",
        weights_path=str(Path(ns.weights)) if merged_given else None,
        backbone_path=str(Path(ns.backbone)) if split_given else None,
        head_path=str(Path(ns.head)) if split_given else None,
        store_path=out_path,
        task_type=task_type,
        derived_transform=derived_transform,
        applied_dtype=applied_dtype,
        saved_logits=save_logits,
        logits_path=str(logits_path) if logits_path is not None else None,
        conform=conform,
        orig_yx=orig_yx,
        conformed_yx=conformed_yx,
        output_scale=output_scale_record,
    )
    print(str(out_path))
    return _EXIT_OK


def _write_predict_store(
    out: Any,  # torch.Tensor at runtime; typed as Any so the module top stays torch-free
    out_path: Path,
    *,
    derived_transform: str | None,
) -> None:
    """Persist the ``infer`` main store to ``out_path``.

    Suffix routing: ``.tif`` / ``.tiff`` → tifffile; ``.npy`` → numpy.save;
    ``.npz`` → numpy.savez.

    Shape contract: ``predict`` returns the tensor in caller-axes order — the
    CLI uses ``axes='yx'`` / ``'zyx'``. ``model.apply_output`` collapses the
    channel axis for both task types (image2image squeeze C=1; image2label
    argmax), so ``[Y, X]`` / ``[Z, Y, X]`` lands here for the canonical output;
    when ``derived_transform is None`` the tensor is logits with a leading C
    axis (``[C, Y, X]`` / ``[C, Z, Y, X]``).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = out.detach().cpu().numpy()
    suffix = out_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        tifffile.imwrite(str(out_path), arr)
    elif suffix == ".npy":
        np.save(out_path, arr)
    elif suffix == ".npz":
        np.savez(out_path, output=arr)
    else:
        raise OmniEMError(
            f"Unsupported output extension {out_path.suffix!r}; use .tif, .npy, or .npz."
        )


def _write_infer_sidecar(
    sidecar_path: Path,
    *,
    model: OmniEM,
    input_path: Path,
    axes: str,
    scale_record: dict[str, Any],
    norm_record: dict[str, Any],
    load_mode: str,
    weights_path: str | None,
    backbone_path: str | None,
    head_path: str | None,
    store_path: Path,
    task_type: str | None,
    derived_transform: str | None,
    applied_dtype: str | None,
    saved_logits: bool,
    logits_path: str | None,
    conform: str,
    orig_yx: list[int],
    conformed_yx: list[int],
    output_scale: dict[str, Any] | None = None,
) -> None:
    """Write a JSON sidecar describing the ``infer`` run.

    Fields:
        * ``arch`` is the MODEL arch (``"omniemv1"``); ``encoder`` is the
          encoder arch (``"emdinov1"``) from the config.
        * ``load_mode`` ∈ ``{merged, split}``; split mode records both paths.
        * ``norm`` records what actually ran: ``{"kind": "config"|"override"|
          "prenormalized", "mean": ..., "std": ...}``.
        * ``task_type`` ∈ ``{None, "image2image", "image2label"}``.
          ``transform`` is the derived output stage (``"image" | "labels" |
          null`` — null when ``task_type`` is None / output = logits).
          ``out_dtype`` is the int width applied (null when ``task_type`` is None);
          ``saved_logits`` is True iff a float-logits sibling was written;
          ``logits_path`` records its path. ``scale`` is the scale-axis record.
        * ``output_scale`` records the ``--output-scale`` pre-resize as
          ``{"factor": F, "input_yx": [Y,X], "scaled_yx": [Y',X']}`` (``null`` when
          the flag was unused).
    """
    cfg = model.config
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "store": store_path.name,
        "input": str(input_path),
        "axes": axes,
        # Explicit task/output fields (replace the old `output` flag).
        "task_type": task_type,
        "transform": derived_transform,
        # out_dtype is null when task_type is None (logits-only output).
        "out_dtype": applied_dtype,
        "saved_logits": bool(saved_logits),
        "logits_path": logits_path,
        # Identifiers: MODEL arch + ENCODER arch — no tag, no unit.
        "arch": cfg.arch,
        "encoder": cfg.encoder,
        "img_z": cfg.img_z,
        "out_channels": cfg.out_channels,
        "load_mode": load_mode,
        "weights": weights_path,
        "backbone": backbone_path,
        "head": head_path,
        "scale": scale_record,
        "norm": norm_record,
        # The conform round-trip — `conform` is the mode applied
        # by apply_input; `orig_yx` is the caller's pre-conform XY; `conformed_yx`
        # is the conformed (potentially padded / resized) XY the net actually saw.
        # `strict` round-trips with orig == conformed so the run is unchanged.
        "conform": conform,
        "orig_yx": orig_yx,
        "conformed_yx": conformed_yx,
        # --output-scale record (always present; `null` when the flag was unused).
        # When set, the input XY was bicubic-resized by `factor` BEFORE the conform
        # round-trip, so the conform `orig_yx` above is the SCALED size; `input_yx`
        # here preserves the true on-disk XY so the full resize is traceable.
        "output_scale": output_scale,
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# --------------------------------------------------------------------------------------
# split / merge — weight-file utilities (config-based load + save_weights round-trip)
# --------------------------------------------------------------------------------------


def _reject_colliding_paths(named: dict[str, str]) -> None:
    """Raise ``OmniEMError`` if any two of the given CLI file paths resolve equal.

    ``named`` maps a flag label (e.g. ``"--backbone"``) to its raw path string. Paths
    are compared by normalized absolute path (``Path(...).resolve()``) so not-yet-existing
    outputs are handled too. This guards split's ``--backbone == --head`` (which would let
    the head write clobber the backbone) and input/output identity (writing an output over
    an input file). The message carries no ``omniem <cmd>:`` prefix — the command wrapper
    adds that.
    """
    seen: dict[Path, str] = {}
    for label, raw in named.items():
        resolved = Path(raw).resolve()
        if resolved in seen:
            raise OmniEMError(
                f"{label} and {seen[resolved]} point to the same path "
                f"({resolved}); they must be distinct."
            )
        seen[resolved] = label


def _cmd_split(ns: argparse.Namespace) -> int:
    """The ``split`` command — merged weight file → backbone + head."""
    try:
        return _run_split(ns)
    except OmniEMError as e:
        print(f"omniem split: {e}", file=sys.stderr)
        return _EXIT_ERROR
    except FileNotFoundError as e:
        print(f"omniem split: file not found: {e}", file=sys.stderr)
        return _EXIT_ERROR
    except OSError as e:  # backstop; the narrow write-time catch (below) is more specific
        print(f"omniem split: {e}", file=sys.stderr)
        return _EXIT_ERROR


def _run_split(ns: argparse.Namespace) -> int:
    # Distinct-path guard first: outputs must differ from each other and from the input.
    _reject_colliding_paths(
        {"--backbone": ns.backbone, "--head": ns.head, "-i/--input": ns.input},
    )
    bb_out = Path(ns.backbone)
    head_out = Path(ns.head)
    if not ns.force:
        for path in (bb_out, head_out):
            if path.exists():
                print(
                    f"omniem split: {path} already exists; pass --force to overwrite.",
                    file=sys.stderr,
                )
                return _EXIT_ERROR

    model = OmniEM.load(Path(ns.model), weights=Path(ns.input))
    try:
        bb_path, head_path = model.save_weights(backbone=bb_out, head=head_out)
    except (OSError, RuntimeError) as e:
        # Narrow write-time catch: name the intended outputs (can't tell which write
        # of the two failed) instead of a bare traceback. torch.save surfaces a bad
        # output path (directory / unwritable) as RuntimeError, not OSError, hence both.
        raise OmniEMError(f"cannot write outputs {bb_out}, {head_out}: {e}") from e
    print(str(bb_path))
    print(str(head_path))
    return _EXIT_OK


def _cmd_merge(ns: argparse.Namespace) -> int:
    """The ``merge`` command — backbone + head → merged weight file."""
    try:
        return _run_merge(ns)
    except OmniEMError as e:
        print(f"omniem merge: {e}", file=sys.stderr)
        return _EXIT_ERROR
    except FileNotFoundError as e:
        print(f"omniem merge: file not found: {e}", file=sys.stderr)
        return _EXIT_ERROR
    except OSError as e:  # backstop; see the narrow write-time catch below
        print(f"omniem merge: {e}", file=sys.stderr)
        return _EXIT_ERROR


def _run_merge(ns: argparse.Namespace) -> int:
    # Distinct-path guard: the merged output must differ from both inputs, and the two
    # inputs from each other.
    _reject_colliding_paths(
        {"--backbone": ns.backbone, "--head": ns.head, "-o/--output": ns.output},
    )
    out = Path(ns.output)
    if not ns.force and out.exists():
        print(
            f"omniem merge: {out} already exists; pass --force to overwrite.",
            file=sys.stderr,
        )
        return _EXIT_ERROR

    model = OmniEM.load(Path(ns.model), backbone=Path(ns.backbone), head=Path(ns.head))
    try:
        out_path = model.save_weights(path=out)
    except (OSError, RuntimeError) as e:
        # torch.save reports a bad output path as RuntimeError, not OSError.
        raise OmniEMError(f"cannot write {out}: {e}") from e
    print(str(out_path))
    return _EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``omniem`` console script.

    Installs a ``warnings.showwarning`` hook for the duration of the command so the
    core's ``OmniEMWarning`` range checks render to stderr as ``omniem <cmd>: …``
    (no raw stdlib formatting, no double-print — the CLI never re-checks ranges).
    Other warning categories keep their default rendering.
    """
    parser = _build_parser()
    ns = parser.parse_args(argv)
    cmd = getattr(ns, "command", None) or "omniem"

    with warnings.catch_warnings():
        warnings.simplefilter("always", OmniEMWarning)

        def _show(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
            if issubclass(category, OmniEMWarning):
                print(f"omniem {cmd}: {message}", file=sys.stderr)
            else:
                sys.stderr.write(
                    warnings.formatwarning(message, category, filename, lineno, line)
                )

        warnings.showwarning = _show
        return ns.func(ns)


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())

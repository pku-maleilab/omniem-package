"""Typed error taxonomy for the ``omniem`` package.

A single ``OmniEMError`` base + flat subclasses. Inheriting from the same base lets
callers ``except OmniEMError`` once to catch every package-raised error; inheriting
each from a stdlib type that already conveys the semantic ((``ValueError`` for input/
config validation, ``RuntimeError`` for runtime contracts, ``ImportError`` for missing
deps, ``MemoryError`` for OOM) keeps third-party code that catches the stdlib type
working without changes.

The taxonomy is intentionally small:

* :class:`ConfigError`         — config schema / version / validation failures
                                 (including missing config files, unknown ``arch``).
* :class:`WeightFormatError`   — checkpoint schema / namespace / meta mismatch
                                 (declared now for a stable taxonomy).
* :class:`MissingExtraError`   — a heavy optional dependency (e.g. monai for ``[infer]``)
                                 is required but absent; ``require_extra()`` raises it.
* :class:`InputContractError`  — caller violated the input contract (wrong axes, wrong
                                 unit, non-square XY, oversize input, …).
* :class:`OOMError`            — explicit OOM signal (declared now for the taxonomy).
"""

from __future__ import annotations


class OmniEMError(Exception):
    """Base for every error raised by the ``omniem`` package.

    Callers can ``except OmniEMError`` to catch any of the typed subclasses at once.
    The subclasses additionally inherit from a stdlib type that already conveys the
    semantic, so generic handlers (``except ValueError``, ``except ImportError`` …)
    still match without needing to know about ``omniem``.
    """


class ConfigError(OmniEMError, ValueError):
    """Configuration validation failure.

    Raised on YAML schema mismatch, schema-version policy violation, missing config
    files, an unknown ``arch`` name, and similar caller-facing config contract
    violations. Inheriting from :class:`ValueError` because the underlying problem
    is "bad data passed in" (whether from YAML or a function argument).
    """


class WeightFormatError(OmniEMError, RuntimeError):
    """Weight file / checkpoint schema or metadata mismatch.

    Used by encoder/head split-file loading. Declared up front so the taxonomy
    is stable from the start.
    """


class MissingExtraError(OmniEMError, ImportError):
    """A heavy optional dependency is missing.

    Raised by :func:`omniem._extras.require_extra`. The message names the extra
    (e.g. ``infer``) and the ``pip install omniem[<extra>]`` hint, so the user can
    fix the install without consulting the source. Inheriting from ``ImportError``
    keeps generic import-error handlers working.

    NOTE: this class is the surface of the extras boundary. The heavy-dep modules
    debut with their code (never as empty installs); the guard mechanism ships
    up front so that code can
    plug in cleanly.
    """


class InputContractError(OmniEMError, ValueError):
    """Caller violated the input contract.

    Raised when the caller passes data that the API can detect as wrong — non-square
    XY, wrong axis string, a 2D image into an ``img_z>1`` model, or a
    size that exceeds the bounded single-shot shape. Inheriting from ``ValueError``
    keeps generic input-validation handlers matching.
    """


class OOMError(OmniEMError, MemoryError):
    """Explicit out-of-memory signal from inference orchestration.

    Used by tiling / streaming. Declared up front so the taxonomy
    is stable from the start.
    """


class OmniEMWarning(UserWarning):
    """Non-fatal advisory from the ``omniem`` input contract.

    Emitted (never raised) by the core/API for **warn-only range checks** — e.g. a
    scaled input or a config / override ``mean``/``std`` that falls outside the
    expected ``[0, 1]`` domain. The run still proceeds; the warning surfaces a
    likely-wrong normalization so the caller can fix the scale/domain.

    Inherits from :class:`UserWarning` (not :class:`OmniEMError`) because it is a
    warning, not an error — ``warnings.simplefilter('error', OmniEMWarning)`` lets
    strict callers (and tests) promote it to an exception when they want to.
    """


__all__ = [
    "ConfigError",
    "InputContractError",
    "MissingExtraError",
    "OOMError",
    "OmniEMError",
    "OmniEMWarning",
    "WeightFormatError",
]

"""Extras-guard helper (roadmap step 0.4 / api-design §4 gate).

Heavy optional dependencies live behind ``omniem[infer]`` / ``omniem[volume]`` /
``omniem[full]``. Those extras are not declared in ``pyproject.toml`` until they
debut *with their code* (never as empty installs), but the
*guard mechanism* is core taxonomy — Stage-2 modules will call :func:`require_extra`
on import / use.

Design notes:

* Use :func:`importlib.util.find_spec` to detect *genuine absence*, not a blanket
  ``try / import``. An installed-but-broken optional dep raises ``ImportError`` when
  imported; that signal must reach the caller, not be silently rebadged as
  "missing extra" (which would send the user down the wrong fix path).

* :func:`find_spec` can itself raise ``ValueError`` / ``ImportError`` for unusual
  package states; those propagate (again: do not mask).

* ``MissingExtraError`` is the only error this module raises. The message names the
  extra and the exact ``pip install`` invocation so the user can fix it without
  reading source.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable

from omniem.errors import MissingExtraError


def require_extra(extra: str, *import_names: str) -> None:
    """Ensure every module name in ``import_names`` is importable.

    Args:
        extra: The pyproject extra that provides the missing dependency (e.g.
            ``"infer"``). Used to construct the ``pip install omniem[<extra>]``
            hint in the error message.
        *import_names: Top-level module names that *must* be present (e.g.
            ``"monai"``, ``"dask"``). At least one is required.

    Raises:
        MissingExtraError: If any of the named modules is absent — i.e.
            :func:`importlib.util.find_spec` returns ``None``. The message names
            both the extra and the missing module, and lists the install command.
        ImportError: Re-raised verbatim if ``find_spec`` itself fails for any
            module — that indicates an installed-but-broken dependency, which is
            a different problem the caller must see.

    Example:
        >>> require_extra("infer", "monai")   # raises if monai is not importable
    """
    if not import_names:
        # Programming error in the caller, not a user-facing condition.
        raise ValueError("require_extra() needs at least one module name")

    missing = _find_missing(import_names)
    if missing:
        # Single, clear message with the install command spelled out. Listing every
        # missing module up front (rather than failing on the first one) means a
        # user with multiple missing pieces gets the full picture in one go.
        missing_str = ", ".join(missing)
        raise MissingExtraError(
            f"Optional dependency missing for omniem[{extra}]: "
            f"{missing_str}. Install with: pip install 'omniem[{extra}]'."
        )


def _find_missing(import_names: Iterable[str]) -> list[str]:
    """Return the subset of ``import_names`` whose module cannot be found.

    Wraps :func:`importlib.util.find_spec` to distinguish:

    * ``spec is None``         → genuinely absent → record as missing.
    * ``ModuleNotFoundError``  → also absent (raised when a parent package is
                                 missing); record as missing.
    * any other ``ImportError`` from ``find_spec`` (installed but broken) →
                                 propagate so the caller sees the real failure.

    We do NOT catch ``Exception`` broadly: that would re-introduce the masking
    behavior the extras boundary explicitly forbids.
    """
    missing: list[str] = []
    for name in import_names:
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            # A parent package is absent; treat the target as missing too.
            missing.append(name)
            continue
        # Other ImportError subclasses (installed-but-broken dep) propagate.
        if spec is None:
            missing.append(name)
    return missing


__all__ = ["require_extra"]

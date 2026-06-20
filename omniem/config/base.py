"""BaseConfig (Pydantic v2) with YAML round-trip and version policy.

Pinned semantics:

* ``schema_version`` is a ``str`` in ``"MAJOR.MINOR"`` form.
* The package's current version is the module constant :data:`SCHEMA_VERSION`
  (``"1.0"`` initially; bumped as the schema evolves).
* On load — :meth:`BaseConfig.from_yaml` / :meth:`BaseConfig.model_validate`:

  - ``major  != current.major``           → :class:`ConfigError` (incompatible).
  - ``major  == current.major`` and
    ``minor <= current.minor``            → accept; ordered minor migrations apply
                                            (currently identity — no migrations yet).
  - ``major  == current.major`` and
    ``minor >  current.minor``            → :class:`ConfigError` (can't migrate forward).

  The three behaviours to rely on: round-trip equal, major mismatch raises, and
  minor-too-new raises.

YAML is read/written with PyYAML's ``safe_load`` / ``safe_dump`` (Pydantic does not do
YAML itself).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from omniem.errors import ConfigError

# Current package schema version. Bumped (minor) when adding fields; bumped (major) only
# on breaking shape changes. A single source of truth so every concrete config inherits
# the same default and the migration policy can compare against one constant.
#
# Stays at "1.0" by maintainer decision: the breaking ModelConfig change
# (output_nonlinear dropped, task_type added) is surfaced by Pydantic
# ``extra='forbid'`` (an unknown-key ConfigError) rather than by a major
# version mismatch. Pre-release; version stability over strict gating.
SCHEMA_VERSION: str = "1.0"


def _parse_version(version: str) -> tuple[int, int]:
    """Parse ``"MAJOR.MINOR"`` into ``(major, minor)`` ints.

    Pydantic itself only checks the type (``str``); the *shape* (two int parts) is the
    job of this helper, called by both the field validator (rejects bad shape at parse
    time) and the migration gate (uses the integer parts to compare).
    """
    parts = version.split(".")
    if len(parts) != 2:
        raise ConfigError(f"schema_version must be 'MAJOR.MINOR' (got {version!r})")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ConfigError(
            f"schema_version must be 'MAJOR.MINOR' with int parts (got {version!r})"
        ) from e


def _check_version_policy(loaded_version: str) -> None:
    """Apply the schema-version policy.

    Raises:
        ConfigError: If the major doesn't match the current schema, or the minor is
            ahead of the current schema. Older minors are accepted (and would be
            migrated forward by future ordered helpers — identity for now).
    """
    cur_major, cur_minor = _parse_version(SCHEMA_VERSION)
    loaded_major, loaded_minor = _parse_version(loaded_version)

    if loaded_major != cur_major:
        raise ConfigError(
            f"Incompatible schema_version: file is {loaded_version!r}, "
            f"package supports {SCHEMA_VERSION!r} (major mismatch — no migration)."
        )
    if loaded_minor > cur_minor:
        raise ConfigError(
            f"schema_version {loaded_version!r} is newer than this package's "
            f"{SCHEMA_VERSION!r} — please upgrade omniem (cannot migrate forward)."
        )
    # loaded_minor <= cur_minor — accept. Ordered identity migrations would run here as
    # real ones are introduced; there are none yet.


class BaseConfig(BaseModel):
    """Shared root for every omniem config model.

    Pydantic-v2 features used:

    * ``ConfigDict(extra="forbid")`` — unknown YAML keys raise on load, which catches
      typos and stale-field regressions early. Concrete configs can relax this in
      later phases if a schema_version migration legitimately introduces an alias.

    * ``field_validator("schema_version")`` — fast shape check at parse time so that
      a malformed version string fails before the policy gate even runs.

    Subclasses inherit ``schema_version`` and the load/dump helpers; they only need to
    declare their own fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        # Validation runs even when the user constructs the model directly (not just
        # via model_validate), so `BaseConfig(schema_version="bad")` also raises.
        validate_assignment=True,
    )

    # All configs carry the same field so the version policy is uniform across
    # encoders/heads/models. Default = current version so freshly-constructed in-memory
    # configs round-trip without the caller having to set it.
    schema_version: str = SCHEMA_VERSION

    # ---- validators ----------------------------------------------------------------

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version_shape(cls, v: str) -> str:
        """Reject malformed strings at parse time (before the policy gate runs)."""
        _parse_version(v)
        return v

    # ---- YAML I/O ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, source: str | Path) -> BaseConfig:
        """Load a config from a YAML file path or a YAML string.

        Args:
            source: A filesystem path (``str`` / ``Path``) OR a literal YAML string.
                We treat any input that exists as a file as a path; otherwise it's
                parsed as inline YAML. This matches the api-design "name_or_cfg"
                resolution rule for ``EMEncoder.load`` / ``OmniEM.load`` (a path-like
                thing that is an existing file wins; everything else is inline).

        Raises:
            ConfigError: On YAML parse failure, missing required keys, or any
                schema-version policy violation.
        """
        data = _load_yaml_source(source)
        return cls.model_validate(data)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> BaseConfig:  # type: ignore[override]
        """Pydantic's hook + the version-policy gate.

        We let Pydantic do its normal validation first (which catches missing
        fields, type errors, extras=forbid violations), then apply the schema gate.
        Wrapping the version check here (not in a ``model_validator``) means
        constructing a model in-memory with the current version never hits the gate
        at all — only loaded data is checked, which is the only place stale versions
        can come from.
        """
        try:
            instance = super().model_validate(obj, **kwargs)
        except ConfigError:
            raise
        except Exception as e:
            # Wrap Pydantic's ValidationError (and anything else from validation) in
            # our taxonomy so callers can `except ConfigError` uniformly. We keep the
            # original cause for debuggability.
            raise ConfigError(str(e)) from e
        _check_version_policy(instance.schema_version)
        return instance

    def to_yaml(self) -> str:
        """Dump to a YAML string (round-trips with :meth:`from_yaml`).

        Uses ``safe_dump(sort_keys=False)`` so the order declared on the model is
        preserved (handy for readable configs); round-trip equality is checked on
        the parsed dict, not on the raw text, so ordering is not load-bearing.
        """
        return yaml.safe_dump(self.model_dump(), sort_keys=False)

    def write_yaml(self, path: str | Path) -> Path:
        """Write the YAML dump to ``path`` and return the resolved path.

        Atomic write would be over-engineering at the config level (configs are
        author-time, not runtime-mutated).
        """
        p = Path(path)
        p.write_text(self.to_yaml(), encoding="utf-8")
        return p


def _load_yaml_source(source: str | Path) -> Any:
    """Resolve ``source`` to parsed YAML data.

    Path-or-string resolution rule:
    a ``Path`` is always treated as a file; a ``str`` is a file iff it exists on disk,
    otherwise it's treated as inline YAML.

    Robustness against long inline YAML: a YAML string can easily
    exceed the OS's ``NAME_MAX`` / ``PATH_MAX`` (e.g. Linux ``NAME_MAX = 255`` per path
    component) once concrete configs accumulate fields. A naive ``Path(s).is_file()``
    raises ``OSError [Errno 36] File name too long`` on such inputs. We:

    1. Early-exit on strings that obviously cannot be filesystem paths — anything
       containing a newline (``\\n`` / ``\\r``) is inline YAML by definition (paths
       don't span lines).
    2. Wrap ``is_file()`` in ``try/except OSError`` so any length / encoding issue from
       the path-probe gets reinterpreted as "this is not a file" → treat as inline YAML.
       We do NOT swallow non-OS errors here.
    """
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, str):
        if _looks_like_inline_yaml(source) or not _is_existing_file(source):
            text = source
        else:
            text = Path(source).read_text(encoding="utf-8")
    else:
        # Type-checker should have caught this; defensive only.
        raise TypeError(f"Expected str or Path, got {type(source).__name__}")
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML: {e}") from e


def _looks_like_inline_yaml(s: str) -> bool:
    """Fast pre-check: a string with a newline cannot be a filesystem path."""
    return "\n" in s or "\r" in s


def _is_existing_file(s: str) -> bool:
    """``Path(s).is_file()`` that returns ``False`` instead of raising ``OSError``.

    For inputs longer than the platform's path limits, ``is_file()`` raises ``OSError``
    rather than returning ``False`` — that's fine for us, we just want to know whether
    the string names an existing file or not.
    """
    try:
        return Path(s).is_file()
    except OSError:
        # Filename too long, invalid characters for a path on this platform, etc. →
        # by definition not a file we can open, so treat as inline content.
        return False

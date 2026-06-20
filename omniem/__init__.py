"""omniem — GUI-free inference & utilities for EM-specific encoders and OmniEM models.

Public package surface:

* Typed error taxonomy re-exported from
  :mod:`omniem.errors`.
* :class:`EMEncoder` + :class:`OmniEM`.
* The arch catalog helpers :func:`list_encoders` / :func:`arch_info` (encoders)
  and :func:`list_models` / :func:`model_arch_info` (models). These describe the
  **frozen, owner-maintained** catalog of architectures that ship with the
  package — not a project-local user registry. ``EMEncoder.load`` selects an
  encoder by ``arch`` name; ``OmniEM.load`` selects a model arch via
  :attr:`omniem.config.ModelConfig.arch`.
* :mod:`omniem.config` — Pydantic-v2 :class:`~omniem.config.BaseConfig` + YAML
  helpers + version-policy gate.
* :mod:`omniem.cli` — argparse entry point for the ``omniem`` console script.

The bigger public API surface — :class:`Inferer`, :class:`Exporter` — lands in
later releases (Stage 2: tiling / streaming / feature export).

Core-deps contract: importing this top-level package must succeed with only
torch + numpy + tifffile + pydantic + pyyaml + **monai** installed. monai became a
**core** dependency: the model decoder reuses MONAI dynunet blocks at module top
level, so monai is imported eagerly (see ``pyproject.toml``). The remaining
heavy/optional deps (dask, h5py, zarr) may NOT be imported at module top level — they
live behind the ``[volume]`` / ``[full]`` extras (which debut **with their code** in
Stage 2) and are guarded at use time via
:func:`omniem._extras.require_extra`.
"""

# Single source of truth for the package version. Kept as a plain string so importing
# omniem never requires importlib.metadata / an installed distribution, and so that
# hatchling can extract it via path-regex without ever importing the package.
__version__ = "0.1.0"


from omniem.encoders import EMEncoder
from omniem.encoders.registry import arch_info, list_encoders
from omniem.errors import (
    ConfigError,
    InputContractError,
    MissingExtraError,
    OmniEMError,
    OmniEMWarning,
    OOMError,
    WeightFormatError,
)
from omniem.models.base import OmniEM
from omniem.models.registry import list_models, model_arch_info

# Public names re-exported from this package.
__all__ = [
    # version
    "__version__",
    # errors + warnings
    "ConfigError",
    "InputContractError",
    "MissingExtraError",
    "OOMError",
    "OmniEMError",
    "OmniEMWarning",
    "WeightFormatError",
    # encoders
    "EMEncoder",
    # models
    "OmniEM",
    # arch catalogs (frozen owner-maintained — encoders + models)
    "arch_info",
    "list_encoders",
    "list_models",
    "model_arch_info",
]

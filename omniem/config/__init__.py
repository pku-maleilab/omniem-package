"""omniem.config — Pydantic v2 + PyYAML config core.

Exposes:

* :class:`BaseConfig` — the shared root model carrying ``schema_version``, with
  YAML load/dump helpers + a version-policy gate.
* :data:`SCHEMA_VERSION` — the package's current schema version (``"MAJOR.MINOR"``).

Concrete configs (``EncoderConfig`` in 1.3, ``HeadConfig`` in 2.1, ``ModelConfig`` /
``InferConfig`` / ``IOConfig`` in 2.5+) all inherit from :class:`BaseConfig` and pick
up the version gate for free.
"""

from omniem.config.base import SCHEMA_VERSION, BaseConfig
from omniem.config.model import ModelConfig

__all__ = ["SCHEMA_VERSION", "BaseConfig", "ModelConfig"]

"""Manifest schema and validation helpers for Experiment 1.

Keep this package initializer lightweight: Blender's Python environment may not
have pandas installed, but Blender-side modules still need schema constants such
as ``VALID_TEXTURE_CONDITIONS``. Pandas-backed manifest helpers are loaded
lazily through ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from exp1.metadata.schema import (
    REQUIRED_RENDER_COLUMNS,
    RENDER_ID_FIELDS,
    VALID_SPLITS,
    VALID_TEXTURE_CONDITIONS,
    generate_render_id,
)

if TYPE_CHECKING:
    from exp1.metadata.manifest import (
        ManifestValidationError,
        attach_render_ids,
        load_manifest,
        save_manifest,
        validate_render_manifest,
    )

_MANIFEST_EXPORTS = {
    "ManifestValidationError",
    "attach_render_ids",
    "load_manifest",
    "save_manifest",
    "validate_render_manifest",
}

__all__ = [
    "ManifestValidationError",
    "REQUIRED_RENDER_COLUMNS",
    "RENDER_ID_FIELDS",
    "VALID_SPLITS",
    "VALID_TEXTURE_CONDITIONS",
    "attach_render_ids",
    "generate_render_id",
    "load_manifest",
    "save_manifest",
    "validate_render_manifest",
]


def __getattr__(name: str):
    if name in _MANIFEST_EXPORTS:
        manifest = import_module("exp1.metadata.manifest")
        return getattr(manifest, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

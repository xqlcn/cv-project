"""Manifest schema and validation helpers for Experiment 1."""

from exp1.metadata.manifest import (
    ManifestValidationError,
    attach_render_ids,
    load_manifest,
    save_manifest,
    validate_render_manifest,
)
from exp1.metadata.schema import (
    REQUIRED_RENDER_COLUMNS,
    RENDER_ID_FIELDS,
    VALID_SPLITS,
    VALID_TEXTURE_CONDITIONS,
    generate_render_id,
)

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


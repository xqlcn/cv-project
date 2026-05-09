"""Asset discovery, normalization, and validation for Experiment 1."""

from exp1.assets.discover import (
    ALLOWED_MESH_EXTENSIONS,
    discover_assets_from_directory,
    discover_modelnet40_assets,
    load_assets_from_manifest,
    standardize_asset_record,
)
from exp1.assets.normalize import normalize_asset_manifest, normalize_asset_record
from exp1.assets.validate import (
    AssetValidationError,
    build_object_split_manifest,
    validate_asset_manifest,
    write_object_split_manifest,
)

__all__ = [
    "ALLOWED_MESH_EXTENSIONS",
    "AssetValidationError",
    "build_object_split_manifest",
    "discover_assets_from_directory",
    "discover_modelnet40_assets",
    "load_assets_from_manifest",
    "normalize_asset_manifest",
    "normalize_asset_record",
    "standardize_asset_record",
    "validate_asset_manifest",
    "write_object_split_manifest",
]


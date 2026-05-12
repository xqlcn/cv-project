"""Asset discovery, normalization, and validation for Experiment 1."""

from exp1.assets.discover import (
    ALLOWED_MESH_EXTENSIONS,
    discover_assets_from_directory,
    discover_modelnet40_assets,
    load_assets_from_manifest,
    standardize_asset_record,
)
from exp1.assets.normalize import normalize_asset_manifest, normalize_asset_record
from exp1.assets.shapenet import (
    DEFAULT_SHAPENET_HF_REPO_ID,
    SHAPENETCORE_GLB_HF_REPO_ID,
    SHAPENETCORE_ZIP_HF_REPO_ID,
    discover_huggingface_shapenet_assets,
    extract_shapenet_archives,
    shapenet_allow_patterns,
)
from exp1.assets.validate import (
    AssetValidationError,
    build_object_split_manifest,
    validate_asset_manifest,
    write_object_split_manifest,
)

__all__ = [
    "ALLOWED_MESH_EXTENSIONS",
    "AssetValidationError",
    "DEFAULT_SHAPENET_HF_REPO_ID",
    "SHAPENETCORE_GLB_HF_REPO_ID",
    "SHAPENETCORE_ZIP_HF_REPO_ID",
    "build_object_split_manifest",
    "discover_assets_from_directory",
    "discover_huggingface_shapenet_assets",
    "discover_modelnet40_assets",
    "extract_shapenet_archives",
    "load_assets_from_manifest",
    "normalize_asset_manifest",
    "normalize_asset_record",
    "shapenet_allow_patterns",
    "standardize_asset_record",
    "validate_asset_manifest",
    "write_object_split_manifest",
]

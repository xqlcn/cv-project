"""Mesh asset discovery for Experiment 1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from exp1.metadata.manifest import load_manifest
from src.datasets.modelnet40_index import discover_modelnet40_records


ALLOWED_MESH_EXTENSIONS = (".obj", ".glb", ".gltf", ".fbx", ".ply", ".off")
SPLIT_ALIASES = {"valid": "val", "validation": "val"}
KNOWN_SPLITS = {"train", "val", "test", "valid", "validation"}
OPTIONAL_ASSET_METADATA_KEYS = (
    "has_photorealistic_material",
    "photorealistic_material_available",
    "has_imported_material",
    "hf_repo_id",
    "hf_revision",
    "shapenet_synset_id",
    "shapenet_model_id",
)


def _clean_split(split: Optional[Any], *, default_split: str = "train") -> str:
    if split is None or str(split).strip() == "":
        return default_split
    split_text = str(split).strip().lower()
    return SPLIT_ALIASES.get(split_text, split_text)


def _infer_split_and_category(
    path: Path,
    root: Path,
    *,
    default_split: str,
) -> tuple[str, str]:
    try:
        rel = path.resolve().relative_to(root.resolve())
        parts = rel.parts
    except ValueError:
        parts = path.parts

    split = default_split
    category = path.parent.name if path.parent.name else "unknown"

    if len(parts) >= 3 and parts[0].lower() in KNOWN_SPLITS:
        split = _clean_split(parts[0], default_split=default_split)
        category = parts[1]
    elif len(parts) >= 3 and parts[1].lower() in KNOWN_SPLITS:
        category = parts[0]
        split = _clean_split(parts[1], default_split=default_split)
    elif len(parts) >= 2:
        category = parts[-2]

    return split, category


def _stable_suffix(value: str, *, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _object_id(category: str, mesh_path: Path, *, used: set[str]) -> str:
    category_text = category or "unknown"
    stem = mesh_path.stem
    base = f"{category_text}_{stem}" if not stem.startswith(category_text) else stem
    oid = base
    if oid in used:
        oid = f"{base}_{_stable_suffix(str(mesh_path.resolve()))}"
    used.add(oid)
    return oid


def standardize_asset_record(
    row: Mapping[str, Any],
    *,
    source_dataset: Optional[str] = None,
    default_split: str = "train",
) -> Dict[str, Any]:
    """Convert existing mesh manifest rows into Experiment 1 asset rows."""
    object_id = row.get("object_id") or row.get("id") or row.get("uid")
    mesh_path = row.get("raw_mesh_path") or row.get("mesh_path") or row.get("path")
    if object_id is None:
        raise ValueError(f"Asset row is missing object_id: {row}")
    if mesh_path is None:
        raise ValueError(f"Asset row for {object_id!r} is missing mesh_path")

    dataset = row.get("source_dataset") or row.get("dataset") or source_dataset
    normalized_path = row.get("normalized_mesh_path") or row.get("normalized_mesh")

    out: Dict[str, Any] = {
        "object_id": str(object_id),
        "source_dataset": str(dataset or "unknown"),
        "category": str(row.get("category", "unknown")),
        "split": _clean_split(row.get("split"), default_split=default_split),
        "raw_mesh_path": str(mesh_path),
        "normalized_mesh_path": str(normalized_path or ""),
        "asset_status": str(row.get("asset_status", "discovered")),
        "asset_error_message": str(row.get("asset_error_message", "")),
    }
    for key in OPTIONAL_ASSET_METADATA_KEYS:
        if key in row:
            out[key] = row[key]
    return out


def discover_assets_from_directory(
    root: Union[str, Path],
    *,
    source_dataset: str,
    allowed_extensions: Sequence[str] = ALLOWED_MESH_EXTENSIONS,
    default_split: str = "train",
    max_objects: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Recursively discover mesh files under a generic asset directory."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Asset root does not exist: {root}")

    allowed = {ext.lower() for ext in allowed_extensions}
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in allowed
    )
    if max_objects is not None:
        paths = paths[: int(max_objects)]

    rows: List[Dict[str, Any]] = []
    used: set[str] = set()
    for mesh_path in paths:
        split, category = _infer_split_and_category(
            mesh_path,
            root,
            default_split=default_split,
        )
        rows.append(
            {
                "object_id": _object_id(category, mesh_path, used=used),
                "source_dataset": source_dataset,
                "category": category,
                "split": split,
                "raw_mesh_path": str(mesh_path),
                "normalized_mesh_path": "",
                "asset_status": "discovered",
                "asset_error_message": "",
            }
        )
    return rows


def discover_modelnet40_assets(
    root: Union[str, Path],
    *,
    splits: Sequence[str] = ("train", "test"),
    max_objects: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Discover ModelNet40 assets using the repo's pure filesystem scanner."""
    records = discover_modelnet40_records(root, splits=splits)
    if max_objects is not None:
        records = records[: int(max_objects)]
    return [
        standardize_asset_record(
            record,
            source_dataset="modelnet40",
            default_split=str(record.get("split", "train")),
        )
        for record in records
    ]


def load_assets_from_manifest(
    path: Union[str, Path],
    *,
    source_dataset: Optional[str] = None,
    default_split: str = "train",
    max_objects: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load a synthetic/ShapeNetCore/Objaverse-style JSON/JSONL/CSV manifest."""
    rows = load_manifest(path, validate=False).to_dict(orient="records")
    if max_objects is not None:
        rows = rows[: int(max_objects)]
    return [
        standardize_asset_record(
            row,
            source_dataset=source_dataset,
            default_split=default_split,
        )
        for row in rows
    ]

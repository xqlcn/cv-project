"""Normalize mesh assets to centered, unit-scale GLB files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Union

import numpy as np


def _safe_component(value: Any) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "unknown"


def _load_trimesh(path: Union[str, Path]):
    import trimesh

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing mesh file: {path}")
    loaded = trimesh.load(path, force=None, process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        if not geoms:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(loaded).__name__} for {path}")
    if loaded.vertices is None or len(loaded.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    return loaded


def _normalize_mesh(mesh: Any, *, target_extent: float) -> tuple[Any, Dict[str, Any]]:
    mesh = mesh.copy()
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Mesh bounds are not finite")

    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    max_extent = float(extents.max())
    if max_extent <= 1e-8:
        raise ValueError(f"Mesh max extent is degenerate: {max_extent}")

    scale = float(target_extent) / max_extent
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) * scale
    normalized_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    normalized_extents = normalized_bounds[1] - normalized_bounds[0]
    metadata = {
        "normalization_center_x": float(center[0]),
        "normalization_center_y": float(center[1]),
        "normalization_center_z": float(center[2]),
        "normalization_scale": scale,
        "normalized_extent_x": float(normalized_extents[0]),
        "normalized_extent_y": float(normalized_extents[1]),
        "normalized_extent_z": float(normalized_extents[2]),
        "normalized_max_extent": float(normalized_extents.max()),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)) if mesh.faces is not None else 0,
    }
    return mesh, metadata


def normalized_output_path(
    row: Mapping[str, Any],
    output_root: Union[str, Path],
) -> Path:
    return (
        Path(output_root)
        / _safe_component(row.get("source_dataset"))
        / _safe_component(row.get("split"))
        / _safe_component(row.get("category"))
        / f"{_safe_component(row.get('object_id'))}.glb"
    )


def normalize_asset_record(
    row: Mapping[str, Any],
    *,
    output_root: Union[str, Path],
    target_extent: float = 1.0,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Normalize one asset row; failures are returned as marked rows."""
    out = dict(row)
    raw_path = Path(str(out.get("raw_mesh_path", ""))).expanduser()
    dst = normalized_output_path(out, output_root)
    out["normalized_mesh_path"] = str(dst)

    try:
        mesh = _load_trimesh(raw_path)
        mesh, metadata = _normalize_mesh(mesh, target_extent=target_extent)
        if overwrite or not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(dst)
        out.update(metadata)
        out["asset_status"] = "normalized"
        out["asset_error_message"] = ""
    except Exception as exc:
        out["normalized_mesh_path"] = ""
        out["asset_status"] = "failed"
        out["asset_error_message"] = str(exc)
    return out


def normalize_asset_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    output_root: Union[str, Path],
    target_extent: float = 1.0,
    overwrite: bool = False,
    fail_fast: bool = False,
) -> List[Dict[str, Any]]:
    """Normalize rows, marking invalid meshes instead of aborting the batch."""
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        result = normalize_asset_record(
            row,
            output_root=output_root,
            target_extent=target_extent,
            overwrite=overwrite,
        )
        if fail_fast and result.get("asset_status") == "failed":
            error = result.get("asset_error_message", "normalization failed")
            raise RuntimeError(error)
        normalized.append(result)
    return normalized

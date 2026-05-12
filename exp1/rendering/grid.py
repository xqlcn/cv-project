"""Controlled render-grid generation for Experiment 1."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from exp1.assets.validate import assign_object_disjoint_splits
from exp1.metadata.manifest import load_manifest, validate_render_manifest
from exp1.metadata.schema import VALID_TEXTURE_CONDITIONS, generate_render_id


DEFAULT_OUTPUT_CONTRACT = {
    "rgb_filename": "rgb.png",
    "depth_filename": "depth.npy",
    "normal_filename": "normal_camera.npy",
    "mask_filename": "mask.npy",
}

OPTIONAL_ASSET_METADATA_KEYS = (
    "has_photorealistic_material",
    "photorealistic_material_available",
    "has_imported_material",
    "hf_repo_id",
    "hf_revision",
    "shapenet_synset_id",
    "shapenet_model_id",
)


def _stable_int(parts: Iterable[Any], *, modulo: int = 2**31 - 1) -> int:
    return int(_stable_hex(parts, length=12), 16) % modulo


def _stable_hex(parts: Iterable[Any], *, length: int = 16) -> str:
    encoded = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:length]


def _as_list(values: Any, *, name: str) -> List[Any]:
    if values is None:
        raise ValueError(f"Grid field '{name}' must not be null")
    if isinstance(values, (str, bytes)):
        return [values]
    out = list(values)
    if not out:
        raise ValueError(f"Grid field '{name}' must not be empty")
    return out


def _infer_split_from_path(mesh_path: Optional[str]) -> Optional[str]:
    if not mesh_path:
        return None
    parts = {part.lower() for part in Path(str(mesh_path)).parts}
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return None


def _normalize_split(
    split: Optional[Any],
    mesh_path: Optional[str],
    default_split: str,
) -> str:
    if split is None or str(split).strip() == "":
        inferred = _infer_split_from_path(mesh_path)
        return inferred or default_split
    split_text = str(split).strip().lower()
    if split_text in {"valid", "validation"}:
        return "val"
    return split_text


def _get_first(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value) != "":
            return value
    return None


def normalize_asset_record(
    row: Mapping[str, Any],
    *,
    source_dataset: Optional[str] = None,
    default_split: str = "train",
) -> Dict[str, Any]:
    """Normalize a mesh asset row into render-plan fields."""
    object_id = _get_first(row, ("object_id", "id", "uid"))
    if object_id is None:
        raise ValueError("Asset row is missing object_id")

    mesh_path = _get_first(row, ("mesh_path", "raw_mesh_path", "path"))
    raw_mesh_path = _get_first(row, ("raw_mesh_path", "mesh_path", "path"))
    if raw_mesh_path is None:
        raise ValueError(f"Asset row for object_id={object_id!r} is missing mesh_path")

    normalized_mesh_path = _get_first(
        row,
        (
            "normalized_mesh_path",
            "normalized_mesh",
            "mesh_path",
            "raw_mesh_path",
            "path",
        ),
    )
    dataset = _get_first(row, ("source_dataset", "dataset", "dataset_name"))

    out: Dict[str, Any] = {
        "object_id": str(object_id),
        "source_dataset": str(dataset or source_dataset or "unknown"),
        "category": str(row.get("category", "unknown")),
        "split": _normalize_split(
            row.get("split"),
            str(mesh_path or raw_mesh_path),
            default_split,
        ),
        "raw_mesh_path": str(raw_mesh_path),
        "normalized_mesh_path": str(normalized_mesh_path or raw_mesh_path),
    }
    for key in OPTIONAL_ASSET_METADATA_KEYS:
        if key in row:
            out[key] = row[key]
    return out


def normalize_asset_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_dataset: Optional[str] = None,
    default_split: str = "train",
    max_objects: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Normalize asset rows while preserving input order and split labels."""
    normalized = [
        normalize_asset_record(
            row,
            source_dataset=source_dataset,
            default_split=default_split,
        )
        for row in rows
    ]
    if max_objects is not None:
        normalized = normalized[: int(max_objects)]
    return normalized


def load_asset_manifest(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a lightweight asset manifest without requiring render-plan columns."""
    df = load_manifest(path, validate=False)
    return df.to_dict(orient="records")


def _resolve_output_contract(
    output_contract: Optional[Mapping[str, Any]],
) -> Dict[str, str]:
    merged = dict(DEFAULT_OUTPUT_CONTRACT)
    if output_contract:
        merged.update({str(k): str(v) for k, v in output_contract.items()})
    return merged


def _render_output_paths(
    *,
    render_root: Union[str, Path],
    split: str,
    object_id: str,
    render_id: str,
    output_contract: Mapping[str, str],
) -> Dict[str, str]:
    root = Path(render_root) / split / object_id / render_id
    return {
        "rgb_path": str(root / output_contract["rgb_filename"]),
        "depth_path": str(root / output_contract["depth_filename"]),
        "normal_path": str(root / output_contract["normal_filename"]),
        "mask_path": str(root / output_contract["mask_filename"]),
    }


def build_render_plan(
    asset_rows: Iterable[Mapping[str, Any]],
    *,
    texture_conditions: Sequence[str] = VALID_TEXTURE_CONDITIONS,
    camera_distances: Sequence[float],
    camera_azimuths_deg: Sequence[float],
    camera_elevations_deg: Sequence[float],
    camera_fov_deg: float,
    light_type: str,
    light_azimuths_deg: Sequence[float],
    light_elevations_deg: Sequence[float],
    light_intensities: Sequence[float],
    object_scales: Sequence[float],
    render_root: Union[str, Path],
    output_contract: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
    texture_seed_offset: int = 100000,
    id_version: str = "v1",
    max_objects: Optional[int] = None,
    source_dataset: Optional[str] = None,
    default_split: str = "train",
) -> pd.DataFrame:
    """Expand asset rows into a texture-controlled render plan."""
    textures = [str(t) for t in _as_list(texture_conditions, name="texture_conditions")]
    invalid_textures = [t for t in textures if t not in set(VALID_TEXTURE_CONDITIONS)]
    if invalid_textures:
        raise ValueError(
            "Invalid texture conditions: "
            + ", ".join(invalid_textures)
            + f". Allowed: {', '.join(VALID_TEXTURE_CONDITIONS)}"
        )

    assets = normalize_asset_records(
        asset_rows,
        source_dataset=source_dataset,
        default_split=default_split,
        max_objects=max_objects,
    )
    output_names = _resolve_output_contract(output_contract)
    rows: List[Dict[str, Any]] = []

    setting_iter = itertools.product(
        _as_list(camera_distances, name="camera_distances"),
        _as_list(camera_azimuths_deg, name="camera_azimuths_deg"),
        _as_list(camera_elevations_deg, name="camera_elevations_deg"),
        _as_list(object_scales, name="object_scales"),
        _as_list(light_azimuths_deg, name="light_azimuths_deg"),
        _as_list(light_elevations_deg, name="light_elevations_deg"),
        _as_list(light_intensities, name="light_intensities"),
    )
    settings = list(setting_iter)

    for asset in assets:
        for setting_idx, setting in enumerate(settings):
            (
                camera_distance,
                camera_azimuth,
                camera_elevation,
                object_scale,
                light_azimuth,
                light_elevation,
                light_intensity,
            ) = setting
            render_seed = _stable_int((seed, asset["object_id"], setting_idx, "render"))
            control_group_id = "tcg_" + _stable_hex(
                (
                    seed,
                    asset["object_id"],
                    asset["source_dataset"],
                    setting_idx,
                    camera_distance,
                    camera_azimuth,
                    camera_elevation,
                    object_scale,
                    light_type,
                    light_azimuth,
                    light_elevation,
                    light_intensity,
                )
            )

            for texture_condition in textures:
                texture_seed = _stable_int(
                    (
                        int(seed) + int(texture_seed_offset),
                        asset["object_id"],
                        setting_idx,
                        texture_condition,
                    )
                )
                row: Dict[str, Any] = {
                    **asset,
                    "texture_condition": texture_condition,
                    "texture_seed": int(texture_seed),
                    "camera_distance": float(camera_distance),
                    "camera_azimuth_deg": float(camera_azimuth),
                    "camera_elevation_deg": float(camera_elevation),
                    "camera_fov_deg": float(camera_fov_deg),
                    "object_scale": float(object_scale),
                    "light_type": str(light_type),
                    "light_azimuth_deg": float(light_azimuth),
                    "light_elevation_deg": float(light_elevation),
                    "light_intensity": float(light_intensity),
                    "render_seed": int(render_seed),
                    "render_status": "pending",
                    "qc_error_message": "",
                    "texture_control_group_id": control_group_id,
                    "grid_index": int(setting_idx),
                }
                row["render_id"] = generate_render_id(row, version=id_version)
                row.update(
                    _render_output_paths(
                        render_root=render_root,
                        split=row["split"],
                        object_id=row["object_id"],
                        render_id=row["render_id"],
                        output_contract=output_names,
                    )
                )
                rows.append(row)

    return validate_render_manifest(pd.DataFrame(rows))


def build_render_plan_from_config(
    asset_rows: Iterable[Mapping[str, Any]],
    cfg: Union[DictConfig, Mapping[str, Any]],
    *,
    max_objects: Optional[int] = None,
) -> pd.DataFrame:
    """Build a render plan from an Experiment 1 OmegaConf config."""
    cfg = OmegaConf.create(cfg)
    max_objects_value = max_objects
    if max_objects_value is None:
        max_objects_value = OmegaConf.select(cfg, "assets.max_objects")

    rows = [dict(row) for row in asset_rows]
    if not bool(OmegaConf.select(cfg, "assets.split_from_manifest", default=False)):
        rows = normalize_asset_records(
            rows,
            max_objects=max_objects_value,
        )
        rows = assign_object_disjoint_splits(
            rows,
            fractions=OmegaConf.to_container(
                OmegaConf.select(cfg, "splits.fractions"),
                resolve=True,
            ),
            labels=[
                str(label)
                for label in OmegaConf.select(
                    cfg,
                    "splits.labels",
                    default=("train", "val", "test"),
                )
            ],
            seed=int(OmegaConf.select(cfg, "splits.seed", default=0)),
        )
        max_objects_value = None

    return build_render_plan(
        rows,
        texture_conditions=list(cfg.textures.conditions),
        camera_distances=list(cfg.camera_grid.distances),
        camera_azimuths_deg=list(cfg.camera_grid.azimuths_deg),
        camera_elevations_deg=list(cfg.camera_grid.elevations_deg),
        camera_fov_deg=float(cfg.camera_grid.fov_deg),
        light_type=str(cfg.lighting_grid.light_type),
        light_azimuths_deg=list(cfg.lighting_grid.azimuths_deg),
        light_elevations_deg=list(cfg.lighting_grid.elevations_deg),
        light_intensities=list(cfg.lighting_grid.intensities),
        object_scales=list(cfg.scale_grid["values"]),
        render_root=str(cfg.paths.render_root),
        output_contract=OmegaConf.to_container(
            cfg.render.output_contract,
            resolve=True,
        ),
        seed=int(cfg.render_plan.seed),
        texture_seed_offset=int(cfg.textures.random_noise.seed_offset),
        id_version=str(cfg.render_plan.id_version),
        max_objects=max_objects_value,
    )

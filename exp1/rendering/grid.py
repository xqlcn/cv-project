"""Controlled render-grid generation for Experiment 1."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from exp1.assets.validate import (
    assign_category_stratified_object_splits,
    assign_object_disjoint_splits,
)
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
    "objaverse_uid",
    "objaverse_cache_bytes",
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


def _as_float_pair(values: Any, *, name: str) -> tuple[float, float]:
    out = _as_list(values, name=name)
    if len(out) != 2:
        raise ValueError(f"Sample range '{name}' must have exactly two values")
    low, high = float(out[0]), float(out[1])
    if high < low:
        raise ValueError(f"Sample range '{name}' must be [low, high], got {out}")
    return low, high


def _sample_uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if low == high:
        return float(low)
    return float(rng.uniform(low, high))


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


def _limit_asset_rows_by_category(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_per_category: Optional[int],
    categories: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    if max_per_category is None and categories is None:
        return [dict(row) for row in rows]
    allowed = None if categories is None else {str(category) for category in categories}
    counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        category = str(row_dict.get("category", "unknown"))
        if allowed is not None and category not in allowed:
            continue
        current = counts.get(category, 0)
        if max_per_category is not None and current >= int(max_per_category):
            continue
        counts[category] = current + 1
        out.append(row_dict)
    return out


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
                    "render_plan_mode": "grid",
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


def build_sampled_render_plan(
    asset_rows: Iterable[Mapping[str, Any]],
    *,
    texture_conditions: Sequence[str] = VALID_TEXTURE_CONDITIONS,
    pose_samples_per_object: int,
    camera_distance: Optional[float] = None,
    camera_distances: Optional[Sequence[float]] = None,
    azimuth_range_deg: Sequence[float] = (0.0, 360.0),
    elevation_range_deg: Sequence[float] = (10.0, 35.0),
    camera_fov_deg: float,
    light_type: str,
    light_azimuth_range_deg: Sequence[float] = (0.0, 360.0),
    light_elevation_range_deg: Sequence[float] = (15.0, 60.0),
    light_intensity_range: Sequence[float] = (2.0, 4.0),
    object_scale_range: Sequence[float] = (0.9, 1.1),
    render_root: Union[str, Path],
    output_contract: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
    texture_seed_offset: int = 100000,
    id_version: str = "v1",
    max_objects: Optional[int] = None,
    source_dataset: Optional[str] = None,
    default_split: str = "train",
) -> pd.DataFrame:
    """Build a deterministic sampled render plan without Cartesian explosion.

    Each object receives ``pose_samples_per_object`` sampled camera/light/scale
    settings. For every setting the function emits matched texture triplets,
    so geometry, pose, lighting, FOV, scale, and render seed are identical
    across ``photorealistic``/``flat``/``random_noise`` rows.
    """
    textures = [str(t) for t in _as_list(texture_conditions, name="texture_conditions")]
    invalid_textures = [t for t in textures if t not in set(VALID_TEXTURE_CONDITIONS)]
    if invalid_textures:
        raise ValueError(
            "Invalid texture conditions: "
            + ", ".join(invalid_textures)
            + f". Allowed: {', '.join(VALID_TEXTURE_CONDITIONS)}"
        )
    samples_per_object = int(pose_samples_per_object)
    if samples_per_object <= 0:
        raise ValueError("pose_samples_per_object must be positive")

    if camera_distances is None:
        if camera_distance is None:
            raise ValueError(
                "Sampled render plans require camera_distance or camera_distances"
            )
        distance_values = [float(camera_distance)]
    else:
        distance_values = [float(v) for v in _as_list(camera_distances, name="camera_distances")]
    if not distance_values:
        raise ValueError("Sampled render plans require at least one camera distance")

    azimuth_bounds = _as_float_pair(azimuth_range_deg, name="azimuth_range_deg")
    elevation_bounds = _as_float_pair(elevation_range_deg, name="elevation_range_deg")
    light_azimuth_bounds = _as_float_pair(
        light_azimuth_range_deg,
        name="light_azimuth_range_deg",
    )
    light_elevation_bounds = _as_float_pair(
        light_elevation_range_deg,
        name="light_elevation_range_deg",
    )
    light_intensity_bounds = _as_float_pair(
        light_intensity_range,
        name="light_intensity_range",
    )
    scale_bounds = _as_float_pair(object_scale_range, name="object_scale_range")

    assets = normalize_asset_records(
        asset_rows,
        source_dataset=source_dataset,
        default_split=default_split,
        max_objects=max_objects,
    )
    output_names = _resolve_output_contract(output_contract)
    rows: List[Dict[str, Any]] = []

    az_low, az_high = azimuth_bounds
    az_span = az_high - az_low
    if az_span <= 0:
        raise ValueError("azimuth_range_deg must have positive width")
    az_bin = az_span / float(samples_per_object)

    for asset in assets:
        for sample_idx in range(samples_per_object):
            rng = random.Random(
                _stable_int(
                    (seed, asset["object_id"], sample_idx, "sampled_setting"),
                    modulo=2**32,
                )
            )
            # Stratify azimuth so each object sees the full view circle without
            # expanding into a dense Cartesian grid.
            camera_azimuth = az_low + (sample_idx + rng.random()) * az_bin
            while camera_azimuth >= az_high:
                camera_azimuth -= az_span
            camera_elevation = _sample_uniform(rng, elevation_bounds)
            object_scale = _sample_uniform(rng, scale_bounds)
            light_azimuth = _sample_uniform(rng, light_azimuth_bounds)
            light_elevation = _sample_uniform(rng, light_elevation_bounds)
            light_intensity = _sample_uniform(rng, light_intensity_bounds)

            for distance_idx, camera_distance_value in enumerate(distance_values):
                setting_idx = sample_idx * len(distance_values) + distance_idx
                render_seed = _stable_int(
                    (seed, asset["object_id"], setting_idx, "render")
                )
                control_group_id = "tcg_" + _stable_hex(
                    (
                        seed,
                        "sampled",
                        asset["object_id"],
                        asset["source_dataset"],
                        setting_idx,
                        camera_distance_value,
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
                        "camera_distance": float(camera_distance_value),
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
                        "pose_sample_index": int(sample_idx),
                        "render_plan_mode": "sampled",
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
    configured_categories = OmegaConf.select(cfg, "assets.categories", default=None)
    configured_max_per_category = OmegaConf.select(
        cfg,
        "assets.max_objects_per_category",
        default=None,
    )
    rows = _limit_asset_rows_by_category(
        rows,
        max_per_category=(
            None
            if configured_max_per_category is None
            else int(configured_max_per_category)
        ),
        categories=(
            None
            if configured_categories is None
            else [str(category) for category in configured_categories]
        ),
    )
    if not bool(OmegaConf.select(cfg, "assets.split_from_manifest", default=False)):
        rows = normalize_asset_records(
            rows,
            max_objects=max_objects_value,
        )
        split_kwargs = {
            "fractions": OmegaConf.to_container(
                OmegaConf.select(cfg, "splits.fractions"),
                resolve=True,
            ),
            "labels": [
                str(label)
                for label in OmegaConf.select(
                    cfg,
                    "splits.labels",
                    default=("train", "val", "test"),
                )
            ],
            "seed": int(OmegaConf.select(cfg, "splits.seed", default=0)),
        }
        if bool(OmegaConf.select(cfg, "splits.stratify_by_category", default=False)):
            rows = assign_category_stratified_object_splits(rows, **split_kwargs)
        else:
            rows = assign_object_disjoint_splits(rows, **split_kwargs)
        max_objects_value = None

    mode = str(OmegaConf.select(cfg, "render_plan.mode", default="grid")).lower()
    if mode == "sampled":
        sampled = OmegaConf.create(
            OmegaConf.select(cfg, "render_plan.sampled", default={}) or {}
        )
        camera_distances = OmegaConf.select(
            sampled,
            "camera_distances",
            default=None,
        )
        camera_distance = OmegaConf.select(
            sampled,
            "camera_distance",
            default=(
                float(cfg.camera_grid.distances[0])
                if len(cfg.camera_grid.distances) > 0
                else None
            ),
        )
        pose_samples = OmegaConf.select(
            cfg,
            "render_plan.pose_samples_per_object",
            default=None,
        )
        if pose_samples is None:
            raise ValueError(
                "render_plan.pose_samples_per_object is required for sampled mode"
            )
        return build_sampled_render_plan(
            rows,
            texture_conditions=list(cfg.textures.conditions),
            pose_samples_per_object=int(pose_samples),
            camera_distance=(
                None if camera_distance is None else float(camera_distance)
            ),
            camera_distances=(
                None
                if camera_distances is None
                else [float(value) for value in camera_distances]
            ),
            azimuth_range_deg=OmegaConf.select(
                sampled,
                "azimuth_range_deg",
                default=[0.0, 360.0],
            ),
            elevation_range_deg=OmegaConf.select(
                sampled,
                "elevation_range_deg",
                default=[
                    min(float(v) for v in cfg.camera_grid.elevations_deg),
                    max(float(v) for v in cfg.camera_grid.elevations_deg),
                ],
            ),
            camera_fov_deg=float(cfg.camera_grid.fov_deg),
            light_type=str(cfg.lighting_grid.light_type),
            light_azimuth_range_deg=OmegaConf.select(
                sampled,
                "light_azimuth_range_deg",
                default=[0.0, 360.0],
            ),
            light_elevation_range_deg=OmegaConf.select(
                sampled,
                "light_elevation_range_deg",
                default=[
                    min(float(v) for v in cfg.lighting_grid.elevations_deg),
                    max(float(v) for v in cfg.lighting_grid.elevations_deg),
                ],
            ),
            light_intensity_range=OmegaConf.select(
                sampled,
                "light_intensity_range",
                default=[
                    min(float(v) for v in cfg.lighting_grid.intensities),
                    max(float(v) for v in cfg.lighting_grid.intensities),
                ],
            ),
            object_scale_range=OmegaConf.select(
                sampled,
                "object_scale_range",
                default=[
                    min(float(v) for v in cfg.scale_grid["values"]),
                    max(float(v) for v in cfg.scale_grid["values"]),
                ],
            ),
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
    if mode != "grid":
        raise ValueError(f"Unsupported render_plan.mode: {mode}")

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

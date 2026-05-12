#!/usr/bin/env python3
"""Estimate Experiment 1 render counts and storage before launching Blender."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.metadata.manifest import load_manifest


def _len_cfg(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (str, bytes)):
        return 1
    return len(list(value))


def _resolve_optional(project_root: Path, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    resolved = resolve_path(project_root, str(path))
    assert resolved is not None
    return resolved


def _configured_object_count(cfg) -> Optional[int]:
    value = cfg.assets.get("max_objects")
    if value is not None:
        return int(value)
    source_limits = [
        int(source.get("max_objects"))
        for source in cfg.assets.sources
        if bool(source.get("enabled", False)) and source.get("max_objects") is not None
    ]
    if source_limits:
        return sum(source_limits)
    return None


def _manifest_object_count(path: Path) -> int:
    rows = load_manifest(path, validate=False)
    if "object_id" not in rows.columns:
        return len(rows)
    return int(rows["object_id"].nunique())


def estimate_exp1_run(
    cfg,
    *,
    asset_manifest: Optional[Path] = None,
    max_objects: Optional[int] = None,
) -> dict[str, Any]:
    """Return render and storage estimates for an Experiment 1 config."""
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    manifest = _resolve_optional(project_root, asset_manifest)
    if manifest is not None and manifest.is_file():
        object_count = _manifest_object_count(manifest)
        object_count_source = str(manifest)
    elif max_objects is not None:
        object_count = int(max_objects)
        object_count_source = "--max-objects"
    else:
        configured = _configured_object_count(cfg)
        object_count = int(configured) if configured is not None else 0
        object_count_source = "config.assets.max_objects"

    texture_count = _len_cfg(cfg.textures.conditions)
    settings_per_object = (
        texture_count
        * _len_cfg(cfg.camera_grid.distances)
        * _len_cfg(cfg.camera_grid.azimuths_deg)
        * _len_cfg(cfg.camera_grid.elevations_deg)
        * _len_cfg(cfg.scale_grid["values"])
        * _len_cfg(cfg.lighting_grid.azimuths_deg)
        * _len_cfg(cfg.lighting_grid.elevations_deg)
        * _len_cfg(cfg.lighting_grid.intensities)
    )
    render_count = int(object_count * settings_per_object)
    width, height = [int(value) for value in cfg.render.resolution]
    pixels = width * height

    # Conservative on-disk estimate for PNG RGB plus NPY depth/normal/mask.
    rgb_bytes = pixels * 3
    depth_bytes = pixels * 4
    normal_bytes = pixels * 3 * 4
    mask_bytes = pixels
    metadata_bytes = 8 * 1024
    random_texture_pixels = int(cfg.textures.random_noise.texture_size[0]) * int(
        cfg.textures.random_noise.texture_size[1]
    )
    random_texture_bytes = random_texture_pixels * 4
    random_noise_fraction = 1.0 / max(texture_count, 1)
    bytes_per_render = (
        rgb_bytes
        + depth_bytes
        + normal_bytes
        + mask_bytes
        + metadata_bytes
        + int(random_texture_bytes * random_noise_fraction)
    )
    render_storage_bytes = int(render_count * bytes_per_render)

    model_dims = {
        "clip_vit_b16": {"final": 512, "default": 768},
        "clip_vit_l14": {"final": 768, "default": 1024},
        "dinov2_vit_b": {"final": 768, "default": 768},
        "dinov2_vit_l": {"final": 1024, "default": 1024},
    }
    feature_bytes = 0
    for model in cfg.models.enabled:
        dims = model_dims.get(str(model), {"final": 768, "default": 768})
        for layer in cfg.models.layers:
            dim = dims["final"] if str(layer) == "final" else dims["default"]
            feature_bytes += render_count * dim * 4

    return {
        "config_name": str(cfg.experiment.name),
        "object_count": int(object_count),
        "object_count_source": object_count_source,
        "settings_per_object": int(settings_per_object),
        "render_count": int(render_count),
        "texture_count": int(texture_count),
        "render_resolution": [width, height],
        "enabled_tasks": [str(task) for task in cfg.tasks.enabled],
        "enabled_models": [str(model) for model in cfg.models.enabled],
        "enabled_layers": [str(layer) for layer in cfg.models.layers],
        "estimated_render_storage_gb": render_storage_bytes / 1024**3,
        "estimated_feature_storage_gb": feature_bytes / 1024**3,
        "estimated_total_storage_gb": (render_storage_bytes + feature_bytes) / 1024**3,
        "chunk_size": int(cfg.render.chunk_size),
        "estimated_chunk_count": int(
            (render_count + int(cfg.render.chunk_size) - 1) // int(cfg.render.chunk_size)
        )
        if render_count
        else 0,
    }


def _format_bytes_gb(value: float) -> str:
    return f"{float(value):.2f} GB"


def print_text_report(estimate: dict[str, Any]) -> None:
    print(f"Experiment: {estimate['config_name']}")
    print(f"Objects: {estimate['object_count']} ({estimate['object_count_source']})")
    print(f"Settings per object: {estimate['settings_per_object']}")
    print(f"Total renders: {estimate['render_count']}")
    print(f"Render resolution: {estimate['render_resolution'][0]}x{estimate['render_resolution'][1]}")
    print(f"Render chunks: {estimate['estimated_chunk_count']} at chunk_size={estimate['chunk_size']}")
    print("Tasks: " + ", ".join(estimate["enabled_tasks"]))
    print("Models: " + ", ".join(estimate["enabled_models"]))
    print("Layers: " + ", ".join(estimate["enabled_layers"]))
    print(
        "Estimated render storage: "
        + _format_bytes_gb(estimate["estimated_render_storage_gb"])
    )
    print(
        "Estimated feature storage: "
        + _format_bytes_gb(estimate["estimated_feature_storage_gb"])
    )
    print(
        "Estimated total storage: "
        + _format_bytes_gb(estimate["estimated_total_storage_gb"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--asset-manifest", type=Path, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    estimate = estimate_exp1_run(
        cfg,
        asset_manifest=args.asset_manifest,
        max_objects=args.max_objects,
    )
    if args.json:
        print(json.dumps(estimate, indent=2, sort_keys=True))
    else:
        print_text_report(estimate)


if __name__ == "__main__":
    main()

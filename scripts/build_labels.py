#!/usr/bin/env python3
"""Build Experiment 1 label tables from render manifests and buffers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.metadata.manifest import load_manifest, save_manifest
from exp1.tasks.camera import build_camera_labels
from exp1.tasks.lighting import build_lighting_labels
from exp1.tasks.relative_depth import build_relative_depth_labels
from exp1.tasks.relative_depth import validate_relative_depth_coverage
from exp1.tasks.scale import build_scale_labels
from exp1.tasks.surface_normals import build_surface_normal_aggregate_labels


DEFAULT_TASKS = (
    "surface_normal_aggregate",
    "relative_depth_regions",
    "dense_depth_patches",
    "camera",
    "camera_distance",
    "viewpoint",
    "lighting",
    "lighting_direction",
    "lighting_intensity",
    "scale",
    "apparent_scale",
)
CAMERA_TASKS = {"camera", "camera_distance", "viewpoint"}
LIGHTING_TASKS = {"lighting", "lighting_direction", "lighting_intensity"}
SCALE_TASKS = {"scale", "apparent_scale"}


def _resolve_required(project_root: Path, path: str) -> Path:
    resolved = resolve_path(project_root, path)
    assert resolved is not None
    return resolved


def _eligible_rows(rows):
    if "qc_pass" in rows.columns:
        rows = rows[rows["qc_pass"].astype(bool)]
    if "render_status" in rows.columns:
        rows = rows[rows["render_status"].astype(str).isin(["success", "passed"])]
    return rows.reset_index(drop=True)


def _output_path(labels_dir: Path, task: str, explicit_suffix: str) -> Path:
    return labels_dir / f"labels_{task}{explicit_suffix}"


def _enabled_tasks(
    config_tasks: Iterable[str],
    explicit: Optional[List[str]],
) -> List[str]:
    if explicit:
        return [str(task) for task in explicit]
    return [str(task) for task in config_tasks]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument(
        "--render-manifest",
        type=Path,
        default=None,
        help="QC/valid render manifest. Defaults to paths.valid_render_manifest.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory for label tables. Defaults to paths.labels_dir.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=DEFAULT_TASKS,
        help="Tasks to build. Defaults to tasks.enabled from config.",
    )
    parser.add_argument(
        "--suffix",
        default=".parquet",
        help="Output suffix: .parquet, .jsonl, .json, or .csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    render_manifest = (
        args.render_manifest
        if args.render_manifest is not None
        else _resolve_required(project_root, str(cfg.paths.valid_render_manifest))
    )
    labels_dir = (
        args.labels_dir
        if args.labels_dir is not None
        else _resolve_required(project_root, str(cfg.paths.labels_dir))
    )
    if not render_manifest.is_absolute():
        render_manifest = project_root / render_manifest
    if not labels_dir.is_absolute():
        labels_dir = project_root / labels_dir

    rows = _eligible_rows(load_manifest(render_manifest, validate=False))
    records = rows.to_dict(orient="records")
    tasks = _enabled_tasks(cfg.tasks.enabled, args.tasks)
    definitions = cfg.tasks.definitions

    written = []
    if "surface_normal_aggregate" in tasks:
        task_cfg = definitions.surface_normal_aggregate
        df = build_surface_normal_aggregate_labels(
            records,
            project_root=project_root,
            min_valid_pixels=int(task_cfg.min_valid_pixels),
        )
        path = _output_path(labels_dir, "surface_normal_aggregate", args.suffix)
        written.append(save_manifest(df, path, validate=False))

    if "relative_depth_regions" in tasks:
        task_cfg = definitions.relative_depth_regions
        df = build_relative_depth_labels(
            records,
            project_root=project_root,
            region_pairs=OmegaConf.to_container(task_cfg.region_pairs, resolve=True),
            grid_size=OmegaConf.to_container(task_cfg.grid_size, resolve=True),
            statistic=str(task_cfg.depth_statistic),
            min_valid_fraction_per_region=float(
                task_cfg.min_valid_fraction_per_region
            ),
            min_depth_margin=float(task_cfg.min_depth_margin),
            use_foreground_bbox=bool(task_cfg.get("use_foreground_bbox", False)),
            bbox_padding_fraction=float(
                task_cfg.get("bbox_padding_fraction", 0.0)
            ),
        )
        validation_cfg = task_cfg.get("validation", {}) or {}
        if bool(validation_cfg.get("enabled", False)):
            summary = validate_relative_depth_coverage(
                records,
                df,
                min_active_pairs=int(validation_cfg.get("min_active_pairs", 4)),
                min_valid_examples=int(
                    validation_cfg.get("min_valid_examples", 1)
                ),
            )
            print("Relative-depth coverage by split/texture:")
            print(summary.to_string(index=False))
        path = _output_path(labels_dir, "relative_depth_regions", args.suffix)
        written.append(save_manifest(df, path, validate=False))

    task_set = set(tasks)
    if task_set & CAMERA_TASKS:
        path = _output_path(labels_dir, "camera", args.suffix)
        written.append(
            save_manifest(build_camera_labels(records), path, validate=False)
        )

    if task_set & LIGHTING_TASKS:
        path = _output_path(labels_dir, "lighting", args.suffix)
        written.append(
            save_manifest(build_lighting_labels(records), path, validate=False)
        )

    if task_set & SCALE_TASKS:
        path = _output_path(labels_dir, "scale", args.suffix)
        written.append(
            save_manifest(
                build_scale_labels(records, project_root=project_root),
                path,
                validate=False,
            )
        )

    for path in written:
        print(f"Wrote labels: {path}")


if __name__ == "__main__":
    main()

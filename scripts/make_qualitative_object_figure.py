#!/usr/bin/env python3
"""Render a qualitative probe figure for a chosen object_id (or category).

Reuses the existing qualitative builder by writing a filtered manifest to a
temporary parquet file and pointing the builder at it. No probe retraining or
feature extraction needed; all required predictions must already exist on disk.

Example:

    PYTHONPATH=. python scripts/make_qualitative_object_figure.py \
        --config configs/exp1_main.yaml \
        --category chair airplane \
        --task relative_depth_regions --layer final
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exp1.analysis.qualitative import (  # noqa: E402
    make_dense_depth_qualitative_table,
    make_dense_surface_normal_qualitative_table,
    make_qualitative_probe_table,
)
from exp1.config import (  # noqa: E402
    default_exp1_config_path,
    load_exp1_config,
    resolve_path,
)
from exp1.metadata.manifest import load_manifest  # noqa: E402


MODEL_DISPLAY_NAMES = {
    "clip_vit_b16": "CLIP B/16",
    "clip_vit_l14": "CLIP L/14",
    "dinov2_vit_b": "DINOv2 B",
    "dinov2_vit_l": "DINOv2 L",
}

DENSE_TASKS = {"dense_depth_patches", "dense_surface_normal_patches"}


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")


def _has_full_texture_triplet(
    df: pd.DataFrame,
    textures: Sequence[str],
    valid_render_ids: set[str],
) -> bool:
    """True if the object has at least one render setting with all three textures."""
    if df.empty:
        return False
    rows = df[df["render_id"].astype(str).isin(valid_render_ids)]
    rows = rows[rows["texture_condition"].astype(str).isin(set(textures))]
    if rows.empty:
        return False
    group_cols = [
        col
        for col in (
            "texture_control_group_id",
            "control_group_id",
            "render_setting_id",
        )
        if col in rows.columns
    ]
    if not group_cols:
        group_cols = [
            col
            for col in (
                "split",
                "camera_distance",
                "camera_azimuth_deg",
                "camera_elevation_deg",
                "light_azimuth_deg",
                "light_elevation_deg",
                "light_intensity",
                "object_scale",
            )
            if col in rows.columns
        ]
    for _, group in rows.groupby(group_cols, sort=False):
        if set(group["texture_condition"].astype(str)) >= set(textures):
            return True
    return False


def _pick_object_id(
    manifest: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    *,
    category: str,
    textures: Sequence[str],
    preferred_split: str,
    seed: int,
) -> str:
    """Choose an arbitrary object_id from the requested category with full triplets."""
    if labels is None:
        valid_ids = set(manifest["render_id"].astype(str))
    elif "label_valid" in labels.columns:
        valid_ids = set(
            labels.loc[labels["label_valid"].astype(bool), "render_id"].astype(str)
        )
    else:
        valid_ids = set(labels["render_id"].astype(str))
    rows = manifest[manifest["category"].astype(str) == category].copy()
    if rows.empty:
        raise SystemExit(f"No manifest rows for category={category!r}")
    rows["_split_pref"] = (rows["split"].astype(str) == preferred_split).astype(int)
    candidates = []
    for object_id, group in rows.groupby("object_id", sort=True):
        if _has_full_texture_triplet(group, textures, valid_ids):
            split_score = int(group["_split_pref"].max())
            candidates.append((split_score, str(object_id)))
    if not candidates:
        raise SystemExit(
            f"No {category!r} objects have a complete {tuple(textures)} triplet "
            "with valid labels."
        )
    candidates.sort(key=lambda t: (-t[0], t[1]))
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    candidates.sort(key=lambda t: -t[0])  # keep preferred-split candidates first
    return candidates[0][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        help="Pick an arbitrary object from each category (one figure per category).",
    )
    parser.add_argument(
        "--object-id",
        nargs="+",
        default=None,
        help="Explicit object_id(s); produces one figure per object.",
    )
    parser.add_argument("--task", default="relative_depth_regions")
    parser.add_argument("--layer", default="final")
    parser.add_argument("--preferred-split", default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write figures into; defaults to cfg.paths.figures_dir.",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=None,
        help="Override cfg.paths.probe_output_dir (e.g. point at a *_rerender tree).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override which models to display; default uses cfg.models.enabled "
        "(or cfg.models.dense_enabled for dense tasks).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=200)
    return parser.parse_args()


def _resolve(project_root: Path, value: Optional[str]) -> Path:
    resolved = resolve_path(project_root, str(value))
    assert resolved is not None
    return resolved


def _resolve_label_path(cfg, project_root: Path, task: str) -> Path:
    label_path = cfg.tasks.definitions[task].get("label_path")
    if label_path is None:
        raise SystemExit(f"Task {task!r} does not define label_path in {cfg=}")
    return _resolve(project_root, label_path)


def _render_one(
    *,
    manifest: pd.DataFrame,
    object_id: str,
    task: str,
    layer: str,
    label_path: Optional[Path],
    probe_root: Path,
    project_root: Path,
    output_dir: Path,
    textures: Sequence[str],
    preferred_split: str,
    model_display_order: dict[str, str],
    image_size: int,
) -> Path:
    subset = manifest[manifest["object_id"].astype(str) == object_id].copy()
    if subset.empty:
        raise SystemExit(f"object_id {object_id!r} not present in manifest")
    figure_name = f"qualitative_{task}_{layer}_{_slug(object_id)}.png"
    output_path = output_dir / figure_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        prefix=f"qualmanifest_{_slug(object_id)}_",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subset.to_parquet(tmp_path, index=False)
        if task == "dense_depth_patches":
            produced = make_dense_depth_qualitative_table(
                manifest_path=tmp_path,
                probe_root=probe_root,
                output_path=output_path,
                project_root=project_root,
                model_display_order=model_display_order,
                layer_name=layer,
                textures=tuple(textures),
                preferred_split=preferred_split,
                image_size=image_size,
            )
        elif task == "dense_surface_normal_patches":
            produced = make_dense_surface_normal_qualitative_table(
                manifest_path=tmp_path,
                probe_root=probe_root,
                output_path=output_path,
                project_root=project_root,
                model_display_order=model_display_order,
                layer_name=layer,
                textures=tuple(textures),
                preferred_split=preferred_split,
                image_size=image_size,
            )
        else:
            if label_path is None:
                raise SystemExit(
                    f"Task {task!r} requires labels but none were resolved from config."
                )
            produced = make_qualitative_probe_table(
                task=task,
                manifest_path=tmp_path,
                label_path=label_path,
                probe_root=probe_root,
                output_path=output_path,
                project_root=project_root,
                model_display_order=model_display_order,
                layer_name=layer,
                textures=tuple(textures),
                preferred_split=preferred_split,
                image_size=image_size,
            )
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    if produced is None:
        raise SystemExit(
            f"Builder returned no path for {task}/{layer}; check that predictions "
            f"exist for object_id={object_id} under {probe_root}"
        )
    print(f"Wrote {produced} (object_id={object_id})")
    return Path(produced)


def main() -> None:
    args = parse_args()
    if not args.category and not args.object_id:
        raise SystemExit("Provide --category and/or --object-id (one figure per value).")

    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    manifest_path = _resolve(project_root, str(cfg.paths.valid_render_manifest))
    if args.probe_root is not None:
        probe_root = args.probe_root
        if not probe_root.is_absolute():
            probe_root = project_root / probe_root
    else:
        probe_root = _resolve(project_root, str(cfg.paths.probe_output_dir))
    figures_dir = (
        args.output_dir
        if args.output_dir is not None
        else _resolve(project_root, str(cfg.paths.figures_dir))
    )
    if not figures_dir.is_absolute():
        figures_dir = project_root / figures_dir

    label_path: Optional[Path] = None
    if args.task not in DENSE_TASKS:
        label_path = _resolve_label_path(cfg, project_root, args.task)

    textures = [str(texture) for texture in cfg.textures.conditions]
    if args.models:
        enabled_models = [str(model) for model in args.models]
    elif args.task in DENSE_TASKS:
        dense_enabled = cfg.models.get("dense_enabled") or cfg.models.enabled
        enabled_models = [str(model) for model in dense_enabled]
    else:
        enabled_models = [str(model) for model in cfg.models.enabled]
    model_display_order = {
        model: MODEL_DISPLAY_NAMES.get(model, model) for model in enabled_models
    }

    manifest = load_manifest(manifest_path, validate=False)
    labels = pd.read_parquet(label_path) if label_path is not None else None

    selected_objects: list[str] = []
    if args.object_id:
        for object_id in args.object_id:
            selected_objects.append(str(object_id))
    if args.category:
        for category in args.category:
            object_id = _pick_object_id(
                manifest,
                labels,
                category=str(category),
                textures=textures,
                preferred_split=args.preferred_split,
                seed=args.seed,
            )
            print(f"Selected {category!r} object: {object_id}")
            selected_objects.append(object_id)

    for object_id in selected_objects:
        _render_one(
            manifest=manifest,
            object_id=object_id,
            task=args.task,
            layer=args.layer,
            label_path=label_path,
            probe_root=probe_root,
            project_root=project_root,
            output_dir=figures_dir,
            textures=textures,
            preferred_split=args.preferred_split,
            model_display_order=model_display_order,
            image_size=int(args.image_size),
        )


if __name__ == "__main__":
    main()

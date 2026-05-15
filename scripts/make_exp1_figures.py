#!/usr/bin/env python3
"""Make Experiment 1 summary figures from aggregated result tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.analysis.plots import plot_layerwise_metrics, plot_texture_drops
from exp1.analysis.qualitative import (
    make_dense_depth_qualitative_table,
    make_dense_surface_normal_qualitative_table,
    make_qualitative_probe_tables,
)
from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.evaluation.comparisons import compute_texture_dependence_drops
from exp1.evaluation.metrics import load_results_table


def _resolve_required(project_root: Path, path: str) -> Path:
    resolved = resolve_path(project_root, path)
    assert resolved is not None
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--texture-drops", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", nargs="+", default=None)
    parser.add_argument("--metric", nargs="+", default=None)
    parser.add_argument("--baseline-texture", default="photorealistic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    results_dir = _resolve_required(project_root, str(cfg.paths.results_dir))
    probe_root = _resolve_required(project_root, str(cfg.paths.probe_output_dir))
    manifest_path = _resolve_required(project_root, str(cfg.paths.valid_render_manifest))
    figures_dir = (
        args.output_dir
        if args.output_dir is not None
        else _resolve_required(project_root, str(cfg.paths.figures_dir))
    )
    results_path = args.results or (results_dir / "exp1_results_long.csv")
    drops_path = args.texture_drops or (results_dir / "exp1_texture_drops.csv")
    if not results_path.is_absolute():
        results_path = project_root / results_path
    if not drops_path.is_absolute():
        drops_path = project_root / drops_path
    if not figures_dir.is_absolute():
        figures_dir = project_root / figures_dir

    results = load_results_table(results_path)
    if drops_path.is_file():
        drops = load_results_table(drops_path)
    else:
        drops = compute_texture_dependence_drops(
            results,
            baseline_texture=str(args.baseline_texture),
        )

    paths = []
    paths.extend(
        plot_layerwise_metrics(
            results,
            figures_dir,
            splits=args.split,
            metrics=args.metric,
        )
    )
    paths.extend(plot_texture_drops(drops, figures_dir, splits=args.split))
    task_label_paths = {}
    for task in cfg.tasks.enabled:
        task_name = str(task)
        label_path = cfg.tasks.definitions[task_name].get("label_path")
        if label_path is None:
            continue
        task_label_paths[task_name] = _resolve_required(project_root, str(label_path))
    model_display_names = {
        "clip_vit_b16": "CLIP B/16",
        "clip_vit_l14": "CLIP L/14",
        "dinov2_vit_b": "DINOv2 B",
        "dinov2_vit_l": "DINOv2 L",
    }
    qualitative_model_display = {
        str(model): model_display_names.get(str(model), str(model))
        for model in cfg.models.enabled
    }
    if any(probe_root.glob("**/predictions.csv")):
        paths.extend(
            make_qualitative_probe_tables(
                tasks=[str(task) for task in cfg.tasks.enabled],
                task_label_paths=task_label_paths,
                manifest_path=manifest_path,
                probe_root=probe_root,
                output_dir=figures_dir,
                project_root=project_root,
                model_display_order=qualitative_model_display,
                layer_name="final",
                textures=[str(texture) for texture in cfg.textures.conditions],
                preferred_split="test",
            )
        )
    dense_layers = cfg.models.get("dense_layers")
    qualitative_layers = (
        [str(layer) for layer in dense_layers]
        if dense_layers
        else ["final"]
    )
    for layer_name in qualitative_layers:
        dense_enabled = cfg.models.get("dense_enabled") or cfg.models.enabled
        dense_model_display = {
            str(model): model_display_names.get(str(model), str(model))
            for model in dense_enabled
        }
        try:
            normal_path = make_dense_surface_normal_qualitative_table(
                manifest_path=manifest_path,
                probe_root=probe_root,
                output_path=figures_dir
                / f"qualitative_dense_surface_normal_patches_{layer_name}.png",
                project_root=project_root,
                model_display_order=dense_model_display,
                layer_name=layer_name,
                textures=[str(texture) for texture in cfg.textures.conditions],
                preferred_split="test",
            )
        except (FileNotFoundError, ValueError, KeyError):
            normal_path = None
        if normal_path is not None:
            paths.append(normal_path)
        try:
            dense_path = make_dense_depth_qualitative_table(
                manifest_path=manifest_path,
                probe_root=probe_root,
                output_path=figures_dir
                / f"qualitative_dense_depth_patches_{layer_name}.png",
                project_root=project_root,
                model_display_order=dense_model_display,
                layer_name=layer_name,
                textures=[str(texture) for texture in cfg.textures.conditions],
                preferred_split="test",
            )
        except (FileNotFoundError, ValueError, KeyError):
            dense_path = None
        if dense_path is not None:
            paths.append(dense_path)
    print(f"Wrote {len(paths)} figure(s) to {figures_dir}")
    for path in paths:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()

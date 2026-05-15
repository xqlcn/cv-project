#!/usr/bin/env python3
"""Train all configured dense per-patch depth probes for Experiment 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_dense_depth_probe import _build_train_config

from exp1.config import (
    default_exp1_config_path,
    ensure_local_hf_home,
    load_exp1_config,
    resolve_path,
)
from exp1.features.storage import patch_feature_cache_path
from exp1.probes.train_dense_depth import train_dense_depth_probe


def _texture_jobs(cfg, explicit_texture_condition):
    if explicit_texture_condition:
        return [
            {
                "name": "texture_"
                + "-".join(str(t) for t in explicit_texture_condition),
                "texture_condition": [str(t) for t in explicit_texture_condition],
                "train_texture_condition": None,
                "eval_texture_condition": None,
            }
        ]
    textures = [str(texture) for texture in cfg.textures.conditions]
    base_eval_cfg = cfg.probe.get("evaluation", {})
    dense_cfg = cfg.probe.get("dense_depth", {}) or {}
    dense_eval_cfg = dense_cfg.get("evaluation", {}) or {}
    within_texture = bool(
        dense_eval_cfg.get(
            "within_texture", base_eval_cfg.get("within_texture", False)
        )
    )
    cross_texture = bool(
        dense_eval_cfg.get(
            "cross_texture", base_eval_cfg.get("cross_texture", False)
        )
    )
    jobs = []
    if within_texture:
        for texture in textures:
            jobs.append(
                {
                    "name": f"within_{texture}",
                    "texture_condition": [texture],
                    "train_texture_condition": None,
                    "eval_texture_condition": None,
                }
            )
    if cross_texture:
        for train_texture in textures:
            for eval_texture in textures:
                if train_texture == eval_texture:
                    continue
                jobs.append(
                    {
                        "name": f"train_{train_texture}__test_{eval_texture}",
                        "texture_condition": None,
                        "train_texture_condition": [train_texture],
                        "eval_texture_condition": [eval_texture],
                    }
                )
    if not jobs:
        jobs.append(
            {
                "name": "all_textures",
                "texture_condition": None,
                "train_texture_condition": None,
                "eval_texture_condition": None,
            }
        )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--layers", nargs="+", default=None)
    parser.add_argument("--texture-condition", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-layernorm", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail instead of skipping when patch caches are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    ensure_local_hf_home(cfg)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    feature_dir = resolve_path(project_root, str(cfg.paths.feature_dir))
    manifest_path = resolve_path(project_root, str(cfg.paths.valid_render_manifest))
    base_output = (
        Path(args.output_dir)
        if args.output_dir is not None
        else resolve_path(project_root, str(cfg.paths.probe_output_dir))
    )
    assert feature_dir is not None and manifest_path is not None and base_output is not None

    dense_enabled = cfg.models.get("dense_enabled") or cfg.models.enabled
    models = args.models or [str(model) for model in dense_enabled]
    dense_layers = cfg.models.get("dense_layers")
    layers = args.layers or (
        [str(layer) for layer in dense_layers]
        if dense_layers
        else [str(layer) for layer in cfg.models.layers]
    )
    texture_jobs = _texture_jobs(cfg, args.texture_condition)
    train_cfg = _build_train_config(cfg, args)

    completed = []
    skipped = []
    for model_name in models:
        for layer_name in layers:
            patch_cache = patch_feature_cache_path(
                feature_dir, model_name=model_name, layer_name=layer_name
            )
            if not Path(patch_cache).is_file():
                message = f"Missing patch cache: {patch_cache}"
                if args.fail_on_missing:
                    raise FileNotFoundError(message)
                print(f"Skipping: {message}")
                skipped.append((model_name, layer_name))
                continue
            for texture_job in texture_jobs:
                tag = f"{model_name}/{layer_name}/dense_depth_patches/{texture_job['name']}"
                output_dir = (
                    Path(base_output)
                    / model_name
                    / layer_name
                    / "dense_depth_patches"
                    / texture_job["name"]
                )
                print(f"Training {tag}")
                result = train_dense_depth_probe(
                    patch_cache=patch_cache,
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    model_name=model_name,
                    layer_name=layer_name,
                    texture_condition=texture_job["texture_condition"],
                    train_texture_condition=texture_job["train_texture_condition"],
                    eval_texture_condition=texture_job["eval_texture_condition"],
                    project_root=project_root,
                    config=train_cfg,
                )
                completed.append((tag, result["metrics"]))
                print(f"Finished {tag}: {result['metrics']}")
    print(f"Completed {len(completed)} dense-depth probe(s); skipped {len(skipped)}.")


if __name__ == "__main__":
    main()

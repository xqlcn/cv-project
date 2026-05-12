#!/usr/bin/env python3
"""Train the configured grid of Experiment 1 linear probes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_exp1_probe import train_one_from_config

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.features.storage import feature_cache_path


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
    eval_cfg = cfg.probe.get("evaluation", {})
    within_texture = bool(eval_cfg.get("within_texture", False))
    cross_texture = bool(eval_cfg.get("cross_texture", False))
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
    parser.add_argument("--tasks", nargs="+", default=None)
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
        help="Fail instead of skipping missing feature caches or label files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    feature_dir = resolve_path(project_root, str(cfg.paths.feature_dir))
    assert feature_dir is not None

    models = args.models or [str(model) for model in cfg.models.enabled]
    layers = args.layers or [str(layer) for layer in cfg.models.layers]
    tasks = args.tasks or [str(task) for task in cfg.tasks.enabled]
    texture_jobs = _texture_jobs(cfg, args.texture_condition)
    completed = []
    skipped = []

    for model_name in models:
        for layer_name in layers:
            cache_path = feature_cache_path(
                feature_dir,
                model_name=model_name,
                layer_name=layer_name,
            )
            for task in tasks:
                task_cfg = cfg.tasks.definitions[task]
                configured_label_path = task_cfg.get("label_path")
                if configured_label_path is None:
                    message = (
                        f"Task {task} does not define label_path; skipping probe "
                        "training for this task."
                    )
                    if args.fail_on_missing:
                        raise ValueError(message)
                    print(f"Skipping: {message}")
                    skipped.append((model_name, layer_name, task))
                    continue
                label_path = resolve_path(
                    project_root,
                    str(configured_label_path),
                )
                assert label_path is not None
                missing = [
                    str(path)
                    for path in (cache_path, label_path)
                    if not Path(path).is_file()
                ]
                if missing:
                    message = (
                        f"Missing inputs for {model_name}/{layer_name}/{task}: "
                        + ", ".join(missing)
                    )
                    if args.fail_on_missing:
                        raise FileNotFoundError(message)
                    print(f"Skipping: {message}")
                    skipped.append((model_name, layer_name, task))
                    continue

                for texture_job in texture_jobs:
                    print(
                        "Training "
                        f"{model_name}/{layer_name}/{task}/"
                        f"{texture_job['name']}"
                    )
                    result = train_one_from_config(
                        cfg,
                        task=task,
                        model_name=model_name,
                        layer_name=layer_name,
                        output_dir=args.output_dir,
                        texture_condition=texture_job["texture_condition"],
                        train_texture_condition=texture_job["train_texture_condition"],
                        eval_texture_condition=texture_job["eval_texture_condition"],
                        args=args,
                    )
                    completed.append(
                        (
                            model_name,
                            layer_name,
                            task,
                            texture_job["name"],
                            result["metrics"],
                        )
                    )
                    print(
                        "Finished "
                        f"{model_name}/{layer_name}/{task}/"
                        f"{texture_job['name']}: {result['metrics']}"
                    )

    print(f"Completed {len(completed)} probe(s); skipped {len(skipped)}.")


if __name__ == "__main__":
    main()

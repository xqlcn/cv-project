#!/usr/bin/env python3
"""Train one Experiment 1 linear probe from cached frozen features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.features.storage import feature_cache_path
from exp1.probes.train import ProbeTrainConfig, train_exp1_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--render-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--texture-condition", nargs="+", default=None)
    parser.add_argument("--train-texture-condition", nargs="+", default=None)
    parser.add_argument("--eval-texture-condition", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-layernorm", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true")
    return parser.parse_args()


def _resolve_required(project_root: Path, path: Optional[Any]) -> Path:
    if path is None:
        raise ValueError("Expected a configured path, got None")
    resolved = resolve_path(project_root, str(path))
    assert resolved is not None
    return resolved


def _task_definition(cfg: DictConfig, task: str) -> DictConfig:
    task_defs = cfg.tasks.definitions
    if task not in task_defs:
        raise KeyError(f"Unknown Experiment 1 task in config: {task}")
    return task_defs[task]


def _target_columns(task_cfg: DictConfig) -> Optional[Sequence[str]]:
    columns = task_cfg.get("target_columns")
    if columns is None:
        columns = task_cfg.get("targets")
    if columns is None and task_cfg.get("target") is not None:
        columns = [str(task_cfg.target)]
    if columns is None:
        return None
    return [str(column) for column in columns]


def _label_path_from_task(task_cfg: DictConfig, *, task: str) -> Any:
    label_path = task_cfg.get("label_path")
    if label_path is None:
        raise ValueError(
            f"Task {task!r} does not define label_path. Build labels for this "
            "task and add a label_path before training a probe."
        )
    return label_path


def _texture_slug(values: Sequence[str]) -> str:
    return "-".join(str(value) for value in values)


def _probe_train_config(
    cfg: DictConfig, args: argparse.Namespace, task: str
) -> ProbeTrainConfig:
    early = cfg.probe.get("early_stopping", {})
    loss_name = str(
        cfg.probe.losses.get(task, cfg.probe.losses.get("regression_default", "mse"))
    )
    return ProbeTrainConfig(
        task=task,
        epochs=int(args.epochs if args.epochs is not None else cfg.probe.epochs),
        batch_size=int(
            args.batch_size if args.batch_size is not None else cfg.probe.batch_size
        ),
        lr=float(args.lr if args.lr is not None else cfg.probe.lr),
        weight_decay=float(
            args.weight_decay
            if args.weight_decay is not None
            else cfg.probe.weight_decay
        ),
        seed=int(args.seed if args.seed is not None else cfg.probe.seed),
        device=str(args.device if args.device is not None else cfg.features.device),
        use_layernorm=not bool(args.no_layernorm) and bool(cfg.probe.use_layernorm),
        surface_normal_loss=loss_name,
        early_stopping=(
            not bool(args.no_early_stopping) and bool(early.get("enabled", True))
        ),
        patience=int(early.get("patience", 8)),
        min_delta=float(early.get("min_delta", 0.0)),
    )


def train_one_from_config(
    cfg: DictConfig,
    *,
    task: str,
    model_name: str,
    layer_name: str,
    feature_cache: Optional[Path] = None,
    label_path: Optional[Path] = None,
    render_manifest: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    texture_condition: Optional[Sequence[str]] = None,
    train_texture_condition: Optional[Sequence[str]] = None,
    eval_texture_condition: Optional[Sequence[str]] = None,
    args: Optional[argparse.Namespace] = None,
) -> dict[str, Any]:
    """Resolve configured paths and train one model/layer/task probe."""
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    task_cfg = _task_definition(cfg, task)
    feature_dir = _resolve_required(project_root, cfg.paths.feature_dir)
    resolved_feature_cache = feature_cache or feature_cache_path(
        feature_dir,
        model_name=model_name,
        layer_name=layer_name,
    )
    resolved_label_path = label_path or _resolve_required(
        project_root,
        _label_path_from_task(task_cfg, task=task),
    )
    resolved_manifest = render_manifest or _resolve_required(
        project_root,
        cfg.paths.valid_render_manifest,
    )
    base_output = output_dir or _resolve_required(
        project_root, cfg.paths.probe_output_dir
    )
    resolved_output_dir = Path(base_output) / model_name / layer_name / task
    if texture_condition:
        resolved_output_dir = resolved_output_dir / (
            "texture_" + _texture_slug(texture_condition)
        )
    elif train_texture_condition or eval_texture_condition:
        train_slug = _texture_slug(train_texture_condition or ["all"])
        eval_slug = _texture_slug(eval_texture_condition or ["all"])
        resolved_output_dir = resolved_output_dir / (
            f"train_{train_slug}__test_{eval_slug}"
        )

    train_cfg = (
        _probe_train_config(cfg, args, task)
        if args is not None
        else ProbeTrainConfig(
            task=task,
            epochs=int(cfg.probe.epochs),
            batch_size=int(cfg.probe.batch_size),
            lr=float(cfg.probe.lr),
            weight_decay=float(cfg.probe.weight_decay),
            seed=int(cfg.probe.seed),
            device=str(cfg.features.device),
            use_layernorm=bool(cfg.probe.use_layernorm),
            surface_normal_loss=str(
                cfg.probe.losses.get(
                    task,
                    cfg.probe.losses.get("regression_default", "mse"),
                )
            ),
            early_stopping=bool(cfg.probe.early_stopping.enabled),
            patience=int(cfg.probe.early_stopping.patience),
            min_delta=float(cfg.probe.early_stopping.min_delta),
        )
    )
    return train_exp1_probe(
        feature_cache=resolved_feature_cache,
        label_path=resolved_label_path,
        manifest_path=resolved_manifest,
        output_dir=resolved_output_dir,
        task=task,
        model_name=model_name,
        layer_name=layer_name,
        target_columns=_target_columns(task_cfg),
        texture_condition=texture_condition,
        train_texture_condition=train_texture_condition,
        eval_texture_condition=eval_texture_condition,
        config=train_cfg,
    )


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    if args.texture_condition and (
        args.train_texture_condition or args.eval_texture_condition
    ):
        raise ValueError(
            "--texture-condition cannot be combined with "
            "--train-texture-condition/--eval-texture-condition"
        )
    task = args.task or str(cfg.tasks.enabled[0])
    model_name = args.model or str(cfg.models.enabled[0])
    layer_name = args.layer or str(cfg.models.layers[0])
    result = train_one_from_config(
        cfg,
        task=task,
        model_name=model_name,
        layer_name=layer_name,
        feature_cache=args.feature_cache,
        label_path=args.label_path,
        render_manifest=args.render_manifest,
        output_dir=args.output_dir,
        texture_condition=args.texture_condition,
        train_texture_condition=args.train_texture_condition,
        eval_texture_condition=args.eval_texture_condition,
        args=args,
    )
    print(OmegaConf.to_yaml({"metrics": result["metrics"]}))
    for name, path in result["artifact_paths"].items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()

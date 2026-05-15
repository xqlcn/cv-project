#!/usr/bin/env python3
"""Train one dense per-patch depth probe for Experiment 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import (
    default_exp1_config_path,
    ensure_local_hf_home,
    load_exp1_config,
    resolve_path,
)
from exp1.features.storage import patch_feature_cache_path
from exp1.probes.train_dense_depth import (
    DenseDepthTrainConfig,
    train_dense_depth_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--patch-cache", type=Path, default=None)
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


def _resolve(project_root: Path, path: Optional[str]) -> Path:
    assert path is not None, "Expected a configured path"
    resolved = resolve_path(project_root, str(path))
    assert resolved is not None
    return resolved


def _dense_cfg_section(cfg) -> dict:
    dense_cfg = cfg.probe.get("dense_depth")
    if dense_cfg is None:
        dense_cfg = OmegaConf.create({})
    return dense_cfg


def _build_train_config(cfg, args: argparse.Namespace) -> DenseDepthTrainConfig:
    dense = _dense_cfg_section(cfg)
    early = cfg.probe.get("early_stopping", {})
    return DenseDepthTrainConfig(
        task=str(dense.get("task", "dense_depth_patches")),
        epochs=int(
            args.epochs
            if args.epochs is not None
            else dense.get("epochs", cfg.probe.epochs)
        ),
        batch_size=int(
            args.batch_size
            if args.batch_size is not None
            else dense.get("batch_size", 32)
        ),
        lr=float(
            args.lr if args.lr is not None else dense.get("lr", cfg.probe.lr)
        ),
        weight_decay=float(
            args.weight_decay
            if args.weight_decay is not None
            else dense.get("weight_decay", cfg.probe.weight_decay)
        ),
        seed=int(args.seed if args.seed is not None else cfg.probe.seed),
        device=str(args.device if args.device is not None else cfg.features.device),
        use_layernorm=not bool(args.no_layernorm)
        and bool(dense.get("use_layernorm", cfg.probe.use_layernorm)),
        scheduler=str(dense.get("scheduler", "cosine_warmup")),
        warmup_epochs=float(dense.get("warmup_epochs", 4.0)),
        monitor=str(dense.get("monitor", "val_ssi_l1_mean")),
        feature_mode=str(dense.get("feature_mode", "patch")),
        target_mode=str(dense.get("target_mode", "ssi_depth")),
        min_valid_fraction_per_patch=float(
            dense.get("min_valid_fraction_per_patch", 0.25)
        ),
        depth_statistic=str(dense.get("depth_statistic", "median")),
        early_stopping=not bool(args.no_early_stopping)
        and bool(early.get("enabled", True)),
        patience=int(early.get("patience", 8)),
        min_delta=float(early.get("min_delta", 0.0)),
    )


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    ensure_local_hf_home(cfg)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()

    model_name = args.model or str(cfg.models.enabled[0])
    layer_default = (
        str(cfg.models.get("dense_layers", cfg.models.layers)[0])
    )
    layer_name = args.layer or layer_default

    feature_dir = _resolve(project_root, str(cfg.paths.feature_dir))
    patch_cache = args.patch_cache or patch_feature_cache_path(
        feature_dir, model_name=model_name, layer_name=layer_name
    )
    if not Path(patch_cache).is_absolute():
        patch_cache = project_root / Path(patch_cache)
    manifest_path = args.render_manifest or _resolve(
        project_root, str(cfg.paths.valid_render_manifest)
    )
    base_output = args.output_dir or _resolve(project_root, str(cfg.paths.probe_output_dir))
    texture_slug_parts: Sequence[str]
    if args.texture_condition:
        texture_slug_parts = ["texture_" + "-".join(args.texture_condition)]
    elif args.train_texture_condition or args.eval_texture_condition:
        train_slug = "-".join(args.train_texture_condition or ["all"])
        eval_slug = "-".join(args.eval_texture_condition or ["all"])
        texture_slug_parts = [f"train_{train_slug}__test_{eval_slug}"]
    else:
        texture_slug_parts = ["all_textures"]
    output_dir = (
        Path(base_output)
        / model_name
        / layer_name
        / "dense_depth_patches"
        / texture_slug_parts[0]
    )

    train_cfg = _build_train_config(cfg, args)
    result = train_dense_depth_probe(
        patch_cache=patch_cache,
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_name=model_name,
        layer_name=layer_name,
        texture_condition=args.texture_condition,
        train_texture_condition=args.train_texture_condition,
        eval_texture_condition=args.eval_texture_condition,
        project_root=project_root,
        config=train_cfg,
    )
    print(OmegaConf.to_yaml({"metrics": result["metrics"]}))
    for name, path in result["artifact_paths"].items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()

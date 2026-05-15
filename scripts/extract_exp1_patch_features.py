#!/usr/bin/env python3
"""Extract frozen CLIP/DINOv2 patch-token grids for Experiment 1 renders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from _repo_root import repo_root  # noqa: E402

PROJECT_ROOT = repo_root(__file__)
sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import (
    default_exp1_config_path,
    ensure_local_hf_home,
    load_exp1_config,
    resolve_path,
)
from exp1.features.extract import (
    extract_and_save_patch_caches,
    load_render_rows_for_features,
)


def _resolve_required(project_root: Path, path: str) -> Path:
    resolved = resolve_path(project_root, path)
    assert resolved is not None
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--render-manifest", type=Path, default=None)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help=(
            "Layers to cache patch tokens for. Defaults to dense_layers from the "
            "config (final + layer8)."
        ),
    )
    parser.add_argument("--split", nargs="+", default=None)
    parser.add_argument("--texture-condition", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for patch tokens. Defaults to float16 to halve disk.",
    )
    parser.add_argument(
        "--include-cls",
        dest="include_cls",
        action="store_true",
        help="Store matching CLS/global features in each patch cache.",
    )
    parser.add_argument(
        "--no-include-cls",
        dest="include_cls",
        action="store_false",
        help="Store patch-token grids only.",
    )
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA fp16 autocast.")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable autocast even if features.use_amp is true.",
    )
    parser.set_defaults(include_cls=True)
    return parser.parse_args()


def _device(requested: Optional[str]) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _resolve_layer_list(cfg, args: argparse.Namespace) -> list[str]:
    if args.layers:
        return [str(layer) for layer in args.layers]
    dense_layers = cfg.models.get("dense_layers")
    if dense_layers:
        return [str(layer) for layer in dense_layers]
    return [str(layer) for layer in cfg.models.layers]


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    ensure_local_hf_home(cfg)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()

    render_manifest = (
        args.render_manifest
        if args.render_manifest is not None
        else _resolve_required(project_root, str(cfg.paths.valid_render_manifest))
    )
    feature_dir = (
        args.feature_dir
        if args.feature_dir is not None
        else _resolve_required(project_root, str(cfg.paths.feature_dir))
    )
    if not render_manifest.is_absolute():
        render_manifest = project_root / render_manifest
    if not feature_dir.is_absolute():
        feature_dir = project_root / feature_dir

    rows = load_render_rows_for_features(
        render_manifest,
        project_root=project_root,
        split=args.split,
        texture_condition=args.texture_condition,
        require_qc_pass=not bool(args.allow_unvalidated),
        limit=args.limit,
    ).to_dict(orient="records")
    if not rows:
        raise RuntimeError(f"No render rows selected from {render_manifest}")

    dense_enabled = cfg.models.get("dense_enabled") or cfg.models.enabled
    model_names = args.models or [str(name) for name in dense_enabled]
    layer_names = _resolve_layer_list(cfg, args)
    device = _device(args.device or str(cfg.features.device))
    batch_size = int(
        args.batch_size if args.batch_size is not None else cfg.features.batch_size
    )
    num_workers = int(
        args.num_workers if args.num_workers is not None else cfg.features.num_workers
    )
    use_amp = bool(cfg.features.get("use_amp", False) or args.amp)
    if args.no_amp:
        use_amp = False
    print(
        f"Extracting patch grids for {len(rows)} renders on {device}; "
        f"models={model_names}, layers={layer_names}, batch_size={batch_size}, "
        f"num_workers={num_workers}, dtype={args.dtype}, include_cls={args.include_cls}, "
        f"use_amp={use_amp}"
    )

    written = []
    model_defs = cfg.models.definitions
    for model_name in model_names:
        if model_name not in model_defs:
            raise KeyError(f"Unknown model in config: {model_name}")
        paths = extract_and_save_patch_caches(
            rows,
            model_name=model_name,
            model_cfg=OmegaConf.to_container(model_defs[model_name], resolve=True),
            layer_names=layer_names,
            feature_dir=feature_dir,
            batch_size=batch_size,
            device=device,
            project_root=project_root,
            num_workers=num_workers,
            dtype=str(args.dtype),
            use_amp=use_amp,
            include_cls=bool(args.include_cls),
        )
        written.extend(paths)
        for path in paths:
            print(f"Wrote patch feature cache: {path}")

    print(f"Wrote {len(written)} patch feature cache(s) under {feature_dir}")


if __name__ == "__main__":
    main()

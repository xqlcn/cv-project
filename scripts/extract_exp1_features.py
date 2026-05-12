#!/usr/bin/env python3
"""Extract frozen CLIP/DINOv2 features for Experiment 1 renders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.features.extract import (
    extract_and_save_feature_caches,
    load_render_rows_for_features,
)


def _resolve_required(project_root: Path, path: str) -> Path:
    resolved = resolve_path(project_root, path)
    assert resolved is not None
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument(
        "--render-manifest",
        type=Path,
        default=None,
        help="Valid render manifest. Defaults to paths.valid_render_manifest.",
    )
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--layers", nargs="+", default=None)
    parser.add_argument("--split", nargs="+", default=None)
    parser.add_argument("--texture-condition", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Do not require qc_pass when the manifest has QC columns.",
    )
    return parser.parse_args()


def _device(requested: Optional[str]) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    if bool(cfg.features.get("use_random_augmentations", False)):
        raise RuntimeError(
            "Experiment 1 feature extraction forbids random augmentations"
        )

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

    model_names = args.models or [str(name) for name in cfg.models.enabled]
    layer_names = args.layers or [str(name) for name in cfg.models.layers]
    device = _device(args.device or str(cfg.features.device))
    print(
        f"Extracting {len(rows)} renders on {device} for models={model_names}, "
        f"layers={layer_names}"
    )

    written = []
    model_defs = cfg.models.definitions
    for model_name in model_names:
        if model_name not in model_defs:
            raise KeyError(f"Unknown model in config: {model_name}")
        paths = extract_and_save_feature_caches(
            rows,
            model_name=model_name,
            model_cfg=OmegaConf.to_container(model_defs[model_name], resolve=True),
            layer_names=layer_names,
            feature_dir=feature_dir,
            batch_size=int(cfg.features.batch_size),
            token=str(cfg.features.token),
            device=device,
            project_root=project_root,
            normalize=bool(cfg.features.normalize),
        )
        written.extend(paths)
        for path in paths:
            print(f"Wrote feature cache: {path}")

    print(f"Wrote {len(written)} feature cache(s) under {feature_dir}")


if __name__ == "__main__":
    main()

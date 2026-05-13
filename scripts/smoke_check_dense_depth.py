#!/usr/bin/env python3
"""Smoke-check dense patch-depth inputs and one training step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.features.storage import load_patch_feature_cache
from exp1.metadata.manifest import load_manifest
from exp1.probes.dense_depth import (
    DenseDepthHead,
    DenseDepthHeadConfig,
    dense_depth_metrics,
    ssi_l1_loss,
)
from exp1.tasks.dense_depth import Exp1DenseDepthDataset


def _resolve(project_root: Optional[Path], value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--patch-cache", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--split", default=None)
    parser.add_argument("--texture-condition", nargs="+", default=None)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--feature-mode", choices=("patch", "patch_cls"), default="patch")
    return parser.parse_args()


def _check_render_buffers(rows: pd.DataFrame, *, project_root: Path) -> None:
    for _, row in rows.iterrows():
        rgb_path = _resolve(project_root, row["rgb_path"])
        depth_path = _resolve(project_root, row["depth_path"])
        mask_path = _resolve(project_root, row["mask_path"])
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        depth = np.load(depth_path, allow_pickle=False)
        mask = np.load(mask_path, allow_pickle=False).astype(bool)
        if depth.shape != mask.shape or depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"Shape mismatch for {row['render_id']}: rgb={rgb.shape}, "
                f"depth={depth.shape}, mask={mask.shape}"
            )
        foreground = mask & np.isfinite(depth) & (depth > 0)
        if not foreground.any():
            raise ValueError(f"No finite foreground depth for {row['render_id']}")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    manifest = load_manifest(args.render_manifest, validate=False)
    if args.split is not None:
        manifest = manifest[manifest["split"].astype(str) == str(args.split)]
    if args.texture_condition is not None:
        allowed = {str(value) for value in args.texture_condition}
        manifest = manifest[manifest["texture_condition"].astype(str).isin(allowed)]
    manifest = manifest.head(int(args.max_rows)).reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("No manifest rows selected for dense-depth smoke check")

    required = {"rgb_path", "depth_path", "mask_path", "render_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError("Manifest missing required columns: " + ", ".join(missing))

    _check_render_buffers(manifest, project_root=project_root)
    cache = load_patch_feature_cache(args.patch_cache)
    try:
        print(
            "Patch cache:",
            {
                "rows": int(len(cache["render_ids"])),
                "patch_features": tuple(int(v) for v in cache["patch_features"].shape),
                "cls_features": (
                    None
                    if cache["cls_features"] is None
                    else tuple(int(v) for v in cache["cls_features"].shape)
                ),
                "patch_grid_shape": cache["patch_grid_shape"],
                "feature_dim": cache["feature_dim"],
            },
        )
    finally:
        cache["_npz_handle"].close()

    dataset = Exp1DenseDepthDataset(
        args.patch_cache,
        manifest=manifest,
        project_root=project_root,
        feature_mode=str(args.feature_mode),
    )
    arrays = dataset.materialize_arrays()
    if len(arrays["render_ids"]) == 0:
        raise RuntimeError("No dense-depth rows survived target materialization")

    x = torch.from_numpy(arrays["features"][: min(4, len(arrays["render_ids"]))])
    y = torch.from_numpy(arrays["targets"][: x.shape[0]])
    m = torch.from_numpy(arrays["valid"][: x.shape[0]])
    model = DenseDepthHead(DenseDepthHeadConfig(feature_dim=x.shape[-1]))
    pred = model(x)
    loss = ssi_l1_loss(pred, y, m)
    loss.backward()
    metrics = dense_depth_metrics(pred.detach(), y, m)
    by_texture = arrays["rows"].groupby("texture_condition")["render_id"].count()
    print("Selected rows by texture:")
    print(by_texture.to_string())
    print("One-batch metrics:")
    print({key: metrics[key] for key in sorted(metrics) if key.endswith("_mean")})


if __name__ == "__main__":
    main()

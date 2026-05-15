#!/usr/bin/env python3
"""Percentile-selected qualitative grids for the rerendered dense Experiment 1.

For each dense task (``dense_depth_patches`` and ``dense_surface_normal_patches``)
we pick representative renders (DINO-best, CLIP-best, both-strong, both-weak,
median) from the test split and produce a grid of:

  RGB | GT label | CLIP prediction | DINO prediction | CLIP error | DINO error

Selection is quantitative (not hand-picked), based on per-render Pearson r
for depth or mean angular error for normals.

Inputs are configurable via CLI; the script does not assume the Colab Drive
mount paths recorded in metadata: it loads NPZ files directly from
``<dense-dir>/probes/...`` and RGB/GT from ``data/exp1_dense/renders/...``
unless you override ``--renders-root``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis._common import (  # noqa: E402
    DENSE_TASKS,
    MODEL_DISPLAY,
    TEXTURE_DISPLAY,
    TEXTURE_ORDER,
    get_pyplot,
    load_dense_npz,
    load_manifest,
    save_figure,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/exp1_dense/manifests/render_valid.parquet"),
    )
    parser.add_argument(
        "--renders-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing data/exp1_dense/renders/...",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/exp1_rerender_analysis_figures/qualitative"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--layer-depth", default="layer8")
    parser.add_argument("--layer-normal", default="layer8")
    parser.add_argument("--n-examples", type=int, default=5)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Render asset loading
# ---------------------------------------------------------------------------


def _resolve_path(renders_root: Path, rel_path: str) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return renders_root / p


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def get_render_assets(
    manifest: pd.DataFrame, render_id: str, renders_root: Path
) -> Optional[Dict[str, object]]:
    row = manifest[manifest["render_id"] == render_id]
    if row.empty:
        return None
    row = row.iloc[0]
    try:
        rgb = load_rgb(_resolve_path(renders_root, row["rgb_path"]))
    except Exception as exc:
        print(f"[warn] load rgb failed for {render_id}: {exc}")
        rgb = None
    try:
        depth = load_npy(_resolve_path(renders_root, row["depth_path"]))
    except Exception:
        depth = None
    try:
        normal = load_npy(_resolve_path(renders_root, row["normal_path"]))
    except Exception:
        normal = None
    try:
        mask = load_npy(_resolve_path(renders_root, row["mask_path"]))
    except Exception:
        mask = None
    return dict(
        row=row,
        rgb=rgb,
        depth=depth,
        normal=normal,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _ssi_normalize(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift-and-scale invariant normalization for visualization (Eigen-style).

    Subtracts the masked median and divides by masked mean absolute deviation.
    """
    valid = mask.astype(bool) & np.isfinite(arr)
    if valid.sum() == 0:
        return np.zeros_like(arr)
    vals = arr[valid]
    med = float(np.median(vals))
    mad = float(np.mean(np.abs(vals - med))) + 1e-6
    out = (arr - med) / mad
    out[~valid] = 0.0
    return out


def _patch_to_image(
    patch_array: np.ndarray, mask_full: Optional[np.ndarray], target_shape: Tuple[int, int]
) -> np.ndarray:
    """Upsample (14,14[,C]) → target_shape (H,W[,C]) by nearest-neighbor."""
    h, w = target_shape[:2]
    src = patch_array
    if src.ndim == 2:
        # (Hp, Wp)
        zoom_h = max(1, h // src.shape[0])
        zoom_w = max(1, w // src.shape[1])
        out = np.kron(src, np.ones((zoom_h, zoom_w)))
        out = out[:h, :w]
    elif src.ndim == 3:
        zoom_h = max(1, h // src.shape[0])
        zoom_w = max(1, w // src.shape[1])
        out = np.kron(src, np.ones((zoom_h, zoom_w, 1)))
        out = out[:h, :w, :]
    else:
        raise ValueError(f"Unsupported patch array shape: {src.shape}")
    return out


def _patch_mask_to_full(valid_patch: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Same as patch_to_image but for boolean validity → full-size mask."""
    return _patch_to_image(valid_patch.astype(float), None, target_shape) > 0.5


def visualize_depth_row(
    ax_rgb, ax_gt, ax_clip, ax_dino, ax_err_clip, ax_err_dino,
    assets,
    clip_pred,
    dino_pred,
    gt_patches,
    valid_patches,
    title_left: str,
    title_right: str,
):
    if assets is None:
        for a in (ax_rgb, ax_gt, ax_clip, ax_dino, ax_err_clip, ax_err_dino):
            a.set_axis_off()
        return
    if assets["rgb"] is not None:
        ax_rgb.imshow(assets["rgb"])
    ax_rgb.set_axis_off()
    ax_rgb.set_title(title_left, fontsize=8)
    H, W = (assets["rgb"].shape[0], assets["rgb"].shape[1]) if assets["rgb"] is not None else (224, 224)

    valid_full = _patch_mask_to_full(valid_patches, (H, W))
    gt_full = _patch_to_image(gt_patches, None, (H, W))
    clip_full = _patch_to_image(clip_pred, None, (H, W))
    dino_full = _patch_to_image(dino_pred, None, (H, W))

    gt_n = _ssi_normalize(gt_full, valid_full)
    clip_n = _ssi_normalize(clip_full, valid_full)
    dino_n = _ssi_normalize(dino_full, valid_full)

    vmin, vmax = -2.0, 2.0
    ax_gt.imshow(gt_n, cmap="turbo", vmin=vmin, vmax=vmax)
    ax_gt.set_axis_off(); ax_gt.set_title("GT depth (SSI)", fontsize=8)
    ax_clip.imshow(clip_n, cmap="turbo", vmin=vmin, vmax=vmax)
    ax_clip.set_axis_off(); ax_clip.set_title("CLIP pred (SSI)", fontsize=8)
    ax_dino.imshow(dino_n, cmap="turbo", vmin=vmin, vmax=vmax)
    ax_dino.set_axis_off(); ax_dino.set_title("DINO pred (SSI)", fontsize=8)

    err_clip = np.abs(clip_n - gt_n)
    err_dino = np.abs(dino_n - gt_n)
    err_clip[~valid_full] = 0.0
    err_dino[~valid_full] = 0.0
    ev_max = float(max(np.percentile(err_clip[valid_full], 95) if valid_full.any() else 1.0,
                       np.percentile(err_dino[valid_full], 95) if valid_full.any() else 1.0,
                       1e-6))
    ax_err_clip.imshow(err_clip, cmap="magma", vmin=0, vmax=ev_max)
    ax_err_clip.set_axis_off(); ax_err_clip.set_title("CLIP |err|", fontsize=8)
    ax_err_dino.imshow(err_dino, cmap="magma", vmin=0, vmax=ev_max)
    ax_err_dino.set_axis_off(); ax_err_dino.set_title(f"DINO |err|\n{title_right}", fontsize=8)


def visualize_normal_row(
    ax_rgb, ax_gt, ax_clip, ax_dino, ax_err_clip, ax_err_dino,
    assets,
    clip_pred,
    dino_pred,
    gt_patches,
    valid_patches,
    title_left: str,
    title_right: str,
):
    if assets is None:
        for a in (ax_rgb, ax_gt, ax_clip, ax_dino, ax_err_clip, ax_err_dino):
            a.set_axis_off()
        return
    if assets["rgb"] is not None:
        ax_rgb.imshow(assets["rgb"])
    ax_rgb.set_axis_off()
    ax_rgb.set_title(title_left, fontsize=8)
    H, W = (assets["rgb"].shape[0], assets["rgb"].shape[1]) if assets["rgb"] is not None else (224, 224)
    valid_full = _patch_mask_to_full(valid_patches, (H, W))
    gt_full = _patch_to_image(gt_patches, None, (H, W))
    clip_full = _patch_to_image(clip_pred, None, (H, W))
    dino_full = _patch_to_image(dino_pred, None, (H, W))

    def norm_for_show(arr: np.ndarray) -> np.ndarray:
        n = arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-8)
        img = (n + 1.0) * 0.5
        img[~valid_full] = 0.0
        return np.clip(img, 0, 1)

    ax_gt.imshow(norm_for_show(gt_full))
    ax_gt.set_axis_off(); ax_gt.set_title("GT normal", fontsize=8)
    ax_clip.imshow(norm_for_show(clip_full))
    ax_clip.set_axis_off(); ax_clip.set_title("CLIP pred", fontsize=8)
    ax_dino.imshow(norm_for_show(dino_full))
    ax_dino.set_axis_off(); ax_dino.set_title("DINO pred", fontsize=8)

    # Angular error in degrees
    def ang_err(pred: np.ndarray) -> np.ndarray:
        gtn = gt_full / (np.linalg.norm(gt_full, axis=-1, keepdims=True) + 1e-8)
        pn = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
        cos = np.clip(np.sum(pn * gtn, axis=-1), -1.0, 1.0)
        deg = np.degrees(np.arccos(cos))
        deg[~valid_full] = 0.0
        return deg

    e_clip = ang_err(clip_full)
    e_dino = ang_err(dino_full)
    vmax = 90.0
    ax_err_clip.imshow(e_clip, cmap="magma", vmin=0, vmax=vmax)
    ax_err_clip.set_axis_off(); ax_err_clip.set_title("CLIP ang err", fontsize=8)
    ax_err_dino.imshow(e_dino, cmap="magma", vmin=0, vmax=vmax)
    ax_err_dino.set_axis_off(); ax_err_dino.set_title(f"DINO ang err\n{title_right}", fontsize=8)


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------


def _score_array(data: Dict[str, np.ndarray], task: str) -> np.ndarray:
    preds = data["predictions"]
    targets = data["targets"]
    valid = data["valid"]
    scores = np.full(len(preds), np.nan)
    for i in range(len(preds)):
        v = valid[i].astype(bool)
        if v.sum() < 3:
            continue
        if task == "dense_depth_patches":
            p = preds[i][v]
            t = targets[i][v]
            if np.std(p) < 1e-8 or np.std(t) < 1e-8:
                scores[i] = 0.0
            else:
                scores[i] = float(np.corrcoef(p, t)[0, 1])
        elif task == "dense_surface_normal_patches":
            p = preds[i][v]
            t = targets[i][v]
            pn = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
            tn = t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)
            cos = np.clip(np.sum(pn * tn, axis=-1), -1.0, 1.0)
            scores[i] = float(np.degrees(np.arccos(cos)).mean())
    return scores


def pick_examples(
    clip_data: Dict[str, np.ndarray],
    dino_data: Dict[str, np.ndarray],
    task: str,
    n_examples: int,
) -> List[Tuple[str, int, int, str]]:
    """Return list of (label, clip_idx, dino_idx, render_id)."""
    clip_ids = list(map(str, clip_data["render_ids"]))
    dino_ids = list(map(str, dino_data["render_ids"]))
    clip_idx = {rid: i for i, rid in enumerate(clip_ids)}
    dino_idx = {rid: i for i, rid in enumerate(dino_ids)}
    shared = [rid for rid in clip_ids if rid in dino_idx]
    if not shared:
        return []
    clip_scores = _score_array(clip_data, task)
    dino_scores = _score_array(dino_data, task)
    higher_is_better = task == "dense_depth_patches"
    chosen: List[Tuple[str, int, int, str]] = []
    rids = np.array(shared)
    cs = np.array([clip_scores[clip_idx[r]] for r in rids])
    ds = np.array([dino_scores[dino_idx[r]] for r in rids])
    if higher_is_better:
        delta = ds - cs
    else:
        delta = cs - ds  # negative means DINO better
    valid = ~(np.isnan(cs) | np.isnan(ds))
    rids = rids[valid]
    cs = cs[valid]
    ds = ds[valid]
    delta = delta[valid]
    if len(rids) == 0:
        return []

    def add(label: str, idx: int) -> None:
        rid = str(rids[idx])
        chosen.append((label, clip_idx[rid], dino_idx[rid], rid))

    # Predefined exemplars; pick top n_examples
    candidates: List[Tuple[str, int]] = []
    if higher_is_better:
        candidates.append(("both strong (high r both)", int(np.argmax(cs + ds))))
        candidates.append(("DINO best (high r, low CLIP r)", int(np.argmax(delta))))
        candidates.append(("CLIP best (high CLIP r, low DINO r)", int(np.argmin(delta))))
        candidates.append(("DINO highest abs r", int(np.argmax(ds))))
        candidates.append(("both weak (low r both)", int(np.argmin(cs + ds))))
        # median DINO score
        sorted_d = np.argsort(ds)
        candidates.append(("median DINO r", int(sorted_d[len(sorted_d) // 2])))
    else:
        candidates.append(("both strong (low err both)", int(np.argmin(cs + ds))))
        candidates.append(("DINO best (low DINO err, high CLIP err)", int(np.argmin(delta))))
        candidates.append(("CLIP best (low CLIP err, high DINO err)", int(np.argmax(delta))))
        candidates.append(("DINO lowest abs err", int(np.argmin(ds))))
        candidates.append(("both weak (high err both)", int(np.argmax(cs + ds))))
        sorted_d = np.argsort(ds)
        candidates.append(("median DINO err", int(sorted_d[len(sorted_d) // 2])))

    seen = set()
    for label, idx in candidates:
        rid = str(rids[idx])
        if rid in seen:
            continue
        seen.add(rid)
        add(label, idx)
        if len(chosen) >= n_examples:
            break
    return chosen


# ---------------------------------------------------------------------------
# Plot grid for a task / texture
# ---------------------------------------------------------------------------


def plot_task_texture(
    task: str,
    texture: str,
    layer: str,
    probe_root: Path,
    manifest: pd.DataFrame,
    renders_root: Path,
    out_dir: Path,
    n_examples: int,
) -> None:
    clip_data = load_dense_npz(probe_root, "clip_vit_b16", layer, task, texture, "test")
    dino_data = load_dense_npz(probe_root, "dinov2_vit_b", layer, task, texture, "test")
    if clip_data is None or dino_data is None:
        print(f"[skip] missing data for {task} / {texture} / {layer}")
        return
    examples = pick_examples(clip_data, dino_data, task, n_examples)
    if not examples:
        print(f"[skip] no examples for {task} / {texture}")
        return
    plt = get_pyplot()
    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 6, figsize=(18, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    clip_scores = _score_array(clip_data, task)
    dino_scores = _score_array(dino_data, task)

    for r, (label, c_idx, d_idx, rid) in enumerate(examples):
        assets = get_render_assets(manifest, rid, renders_root)
        # title: include category and texture
        cat = (assets["row"]["category"] if assets is not None else "?")
        cs_val = float(clip_scores[c_idx])
        ds_val = float(dino_scores[d_idx])
        title_left = f"{label}\n{cat} | {rid[:10]}…"
        title_right = (
            f"r: CLIP={cs_val:.2f} | DINO={ds_val:.2f}"
            if task == "dense_depth_patches"
            else f"angerr deg: CLIP={cs_val:.1f} | DINO={ds_val:.1f}"
        )
        if task == "dense_depth_patches":
            visualize_depth_row(
                axes[r, 0], axes[r, 1], axes[r, 2], axes[r, 3], axes[r, 4], axes[r, 5],
                assets,
                clip_data["predictions"][c_idx],
                dino_data["predictions"][d_idx],
                dino_data["targets"][d_idx],  # GT is the same across models
                dino_data["valid"][d_idx],
                title_left,
                title_right,
            )
        else:
            visualize_normal_row(
                axes[r, 0], axes[r, 1], axes[r, 2], axes[r, 3], axes[r, 4], axes[r, 5],
                assets,
                clip_data["predictions"][c_idx],
                dino_data["predictions"][d_idx],
                dino_data["targets"][d_idx],
                dino_data["valid"][d_idx],
                title_left,
                title_right,
            )
    task_short = "depth" if task == "dense_depth_patches" else "normals"
    fig.suptitle(
        f"Qualitative {task_short} (layer={layer}, texture={TEXTURE_DISPLAY[texture]}, split=test)\n"
        f"Predictions and per-render scores are direction-aware. {('Higher r = better.' if task=='dense_depth_patches' else 'Lower angular error = better.')}",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = out_dir / f"qualitative_{task_short}_{layer}_{texture}.png"
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[done] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_root = (args.dense_dir / "probes").resolve()
    manifest = load_manifest(args.manifest)
    if manifest is None:
        raise SystemExit(f"Could not load manifest at {args.manifest}")
    print(f"[load] dense probes: {probe_root}")
    print(f"[load] manifest rows: {len(manifest)}")

    for task, layer in (
        ("dense_depth_patches", args.layer_depth),
        ("dense_surface_normal_patches", args.layer_normal),
    ):
        for tex in TEXTURE_ORDER:
            plot_task_texture(
                task=task,
                texture=tex,
                layer=layer,
                probe_root=probe_root,
                manifest=manifest,
                renders_root=args.renders_root,
                out_dir=out_dir,
                n_examples=args.n_examples,
            )
    print(f"[ok] figures written to: {out_dir}")


if __name__ == "__main__":
    main()

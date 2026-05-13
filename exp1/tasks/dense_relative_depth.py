"""Dense per-patch depth task.

The MVP relative_depth_regions task asks for a 3x3 coarse pair ordering. The
dense variant in this module pools the rendered ``depth.npy`` to the ViT patch
grid (e.g. 14x14 for CLIP-B/16, 16x16 for CLIP-L/14, or 37x37 for DINOv2-B/14)
so a linear probe can predict a depth value per patch token. The primary loss is
scale- and shift-invariant (SSI), while evaluation still reports metric and
scale-invariant depth metrics.

The functions here do *not* persist a separate label file: depth and mask
arrays already live next to every render on disk. The dense dataset loads them
lazily, NaN-aware-pools to the requested patch grid, and yields (P, P) tensors
keyed by ``render_id``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from exp1.features.storage import load_patch_feature_cache


__all__ = [
    "DenseDepthSample",
    "Exp1DenseDepthDataset",
    "align_depth_mask_to_model_input",
    "nanaware_pool",
]


def _resolve_path(project_root: Optional[Union[str, Path]], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


def nanaware_pool(
    array: np.ndarray,
    *,
    grid_rows: int,
    grid_cols: int,
    mask: Optional[np.ndarray] = None,
    min_valid_fraction: float = 0.0,
    statistic: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware pool ``array`` to a ``(grid_rows, grid_cols)`` grid.

    Returns a ``(grid_rows, grid_cols)`` float32 grid of mean depth values and
    a matching bool ``valid`` mask. A cell is considered valid only when enough
    source pixels are finite (and inside ``mask`` when provided).
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"depth/mask pool input must be 2D, got shape {arr.shape}")
    height, width = arr.shape
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")

    if mask is None:
        valid_input = np.isfinite(arr)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != arr.shape:
            raise ValueError(
                f"mask shape {mask_arr.shape} must match array shape {arr.shape}"
            )
        valid_input = mask_arr & np.isfinite(arr)

    if statistic not in {"mean", "median"}:
        raise ValueError(f"Unsupported depth statistic: {statistic}")

    if (
        statistic == "mean"
        and height % grid_rows == 0
        and width % grid_cols == 0
    ):
        return _nanaware_pool_fast(
            arr,
            valid_input,
            grid_rows,
            grid_cols,
            min_valid_fraction=min_valid_fraction,
        )

    row_edges = np.linspace(0, height, grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_cols + 1, dtype=int)
    pooled = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    valid = np.zeros((grid_rows, grid_cols), dtype=bool)
    min_fraction = max(0.0, float(min_valid_fraction))
    for r in range(grid_rows):
        r0, r1 = int(row_edges[r]), int(row_edges[r + 1])
        if r1 <= r0:
            continue
        for c in range(grid_cols):
            c0, c1 = int(col_edges[c]), int(col_edges[c + 1])
            if c1 <= c0:
                continue
            window = arr[r0:r1, c0:c1]
            mask_window = valid_input[r0:r1, c0:c1]
            count = int(mask_window.sum())
            min_count = max(1, int(np.ceil(mask_window.size * min_fraction)))
            if count < min_count:
                continue
            values = window[mask_window]
            pooled[r, c] = (
                float(values.mean())
                if statistic == "mean"
                else float(np.median(values))
            )
            valid[r, c] = True
    return pooled, valid


def _nanaware_pool_fast(
    arr: np.ndarray,
    valid_input: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    *,
    min_valid_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized pool when ``arr`` divides evenly into the grid."""
    height, width = arr.shape
    cell_h = height // grid_rows
    cell_w = width // grid_cols
    reshaped = arr.reshape(grid_rows, cell_h, grid_cols, cell_w)
    mask_reshaped = valid_input.reshape(grid_rows, cell_h, grid_cols, cell_w)
    counts = mask_reshaped.sum(axis=(1, 3)).astype(np.int32)
    masked = np.where(mask_reshaped, reshaped, 0.0)
    sums = masked.sum(axis=(1, 3))
    min_count = max(
        1,
        int(np.ceil(cell_h * cell_w * max(0.0, float(min_valid_fraction)))),
    )
    valid = counts >= min_count
    pooled = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    np.divide(sums, counts, out=pooled, where=valid)
    return pooled.astype(np.float32, copy=False), valid


def _resize_nearest(array: np.ndarray, *, height: int, width: int) -> np.ndarray:
    dtype = array.dtype
    tensor = torch.from_numpy(np.asarray(array)).float()[None, None]
    out = F.interpolate(tensor, size=(int(height), int(width)), mode="nearest")
    return out[0, 0].cpu().numpy().astype(dtype, copy=False)


def align_depth_mask_to_model_input(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    model_input_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the deterministic resize/center-crop geometry used by ViT inputs."""
    depth_arr = np.asarray(depth, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=bool)
    if depth_arr.ndim != 2 or mask_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D depth/mask arrays, got {depth_arr.shape} and {mask_arr.shape}"
        )
    if depth_arr.shape != mask_arr.shape:
        raise ValueError(
            f"depth shape {depth_arr.shape} must match mask shape {mask_arr.shape}"
        )
    if model_input_size is None or int(model_input_size) <= 0:
        return depth_arr, mask_arr

    target = int(model_input_size)
    height, width = depth_arr.shape
    if height == target and width == target:
        return depth_arr, mask_arr

    scale = float(target) / float(min(height, width))
    resized_h = max(target, int(round(height * scale)))
    resized_w = max(target, int(round(width * scale)))
    depth_resized = _resize_nearest(depth_arr, height=resized_h, width=resized_w)
    mask_resized = (
        _resize_nearest(
            mask_arr.astype(np.float32),
            height=resized_h,
            width=resized_w,
        )
        > 0.5
    )

    y0 = max(0, (resized_h - target) // 2)
    x0 = max(0, (resized_w - target) // 2)
    return (
        depth_resized[y0 : y0 + target, x0 : x0 + target],
        mask_resized[y0 : y0 + target, x0 : x0 + target],
    )


@dataclass
class DenseDepthSample:
    """One dense-depth probe example."""

    render_id: str
    features: np.ndarray  # (P, P, D) float32 patch tokens
    target: np.ndarray  # (P, P) float32 pooled depth
    valid: np.ndarray  # (P, P) bool mask


class Exp1DenseDepthDataset(Dataset):
    """Pair patch features with NaN-pooled per-patch depth labels."""

    def __init__(
        self,
        patch_cache: Union[str, Path],
        *,
        manifest: pd.DataFrame,
        split: Optional[Union[str, Sequence[str]]] = None,
        texture_condition: Optional[Union[str, Sequence[str]]] = None,
        project_root: Optional[Union[str, Path]] = None,
        min_valid_patches: int = 4,
        min_valid_fraction_per_patch: float = 0.25,
        depth_statistic: str = "median",
        feature_mode: str = "patch",
        model_input_size: Optional[int] = None,
        cache_targets: bool = True,
        num_workers: int = 0,
    ) -> None:
        if "render_id" not in manifest.columns:
            raise ValueError("Manifest must have a render_id column")
        if "depth_path" not in manifest.columns or "mask_path" not in manifest.columns:
            raise ValueError(
                "Manifest must include depth_path and mask_path columns for the "
                "dense relative-depth task"
            )

        cache = load_patch_feature_cache(patch_cache, mmap_mode="r")
        self._cache = cache
        self._render_ids = cache["render_ids"]
        self._patch_features = cache["patch_features"]
        self._cls_features = cache.get("cls_features")
        self.patch_grid_shape = tuple(int(v) for v in cache["patch_grid_shape"])
        self.patch_feature_dim = int(cache["feature_dim"])
        self.feature_mode = str(feature_mode)
        if self.feature_mode not in {"patch", "patch_cls"}:
            raise ValueError(
                f"Unsupported dense depth feature_mode: {self.feature_mode}"
            )
        self.cls_feature_dim = 0
        if self.feature_mode == "patch_cls":
            if self._cls_features is None:
                raise ValueError(
                    "feature_mode='patch_cls' requires patch caches saved with "
                    "cls_features; rerun scripts/extract_exp1_patch_features.py"
                )
            self.cls_feature_dim = int(self._cls_features.shape[1])
        self.feature_dim = self.patch_feature_dim + self.cls_feature_dim
        self.project_root = project_root
        self.min_valid_fraction_per_patch = float(min_valid_fraction_per_patch)
        self.depth_statistic = str(depth_statistic)
        metadata = dict(cache.get("metadata", {}))
        configured_input_size = model_input_size or metadata.get("model_input_size")
        self.model_input_size = (
            int(configured_input_size)
            if configured_input_size not in {None, "", 0, "0"}
            else None
        )

        manifest = manifest.copy()
        manifest["render_id"] = manifest["render_id"].astype(str)
        if split is not None:
            split_set = {split} if isinstance(split, str) else set(split)
            manifest = manifest[manifest["split"].astype(str).isin(split_set)]
        if texture_condition is not None:
            tex_set = (
                {texture_condition}
                if isinstance(texture_condition, str)
                else set(texture_condition)
            )
            manifest = manifest[
                manifest["texture_condition"].astype(str).isin(tex_set)
            ]
        cache_id_to_idx = {rid: i for i, rid in enumerate(self._render_ids.tolist())}
        manifest = manifest[manifest["render_id"].isin(cache_id_to_idx)]
        manifest = manifest.reset_index(drop=True)

        rows: list[dict[str, Any]] = []
        for _, row in manifest.iterrows():
            rows.append(
                {
                    "render_id": str(row["render_id"]),
                    "depth_path": str(row["depth_path"]),
                    "mask_path": str(row["mask_path"]),
                    "texture_condition": str(row.get("texture_condition", "")),
                    "split": str(row.get("split", "")),
                    "object_id": str(row.get("object_id", "")),
                    "source_dataset": str(row.get("source_dataset", "")),
                    "category": str(row.get("category", "")),
                    "cache_index": int(cache_id_to_idx[str(row["render_id"])]),
                }
            )
        self.rows = pd.DataFrame(rows)
        self._min_valid_patches = int(min_valid_patches)
        self._target_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cache_targets = bool(cache_targets)
        if cache_targets:
            self._prefill_targets(num_workers=num_workers)

    def __len__(self) -> int:
        return len(self.rows)

    def _pool_target(
        self, depth_path: Union[str, Path], mask_path: Union[str, Path]
    ) -> tuple[np.ndarray, np.ndarray]:
        depth = np.load(
            _resolve_path(self.project_root, depth_path), allow_pickle=False
        ).astype(np.float32)
        mask = np.load(
            _resolve_path(self.project_root, mask_path), allow_pickle=False
        ).astype(bool)
        depth, mask = align_depth_mask_to_model_input(
            depth,
            mask,
            model_input_size=self.model_input_size,
        )
        rows, cols = self.patch_grid_shape
        pooled, valid = nanaware_pool(
            depth,
            grid_rows=rows,
            grid_cols=cols,
            mask=mask,
            min_valid_fraction=self.min_valid_fraction_per_patch,
            statistic=self.depth_statistic,
        )
        return pooled, valid

    def _prefill_targets(self, *, num_workers: int) -> None:
        worker_count = max(0, int(num_workers))
        items = list(zip(self.rows["render_id"].tolist(),
                         self.rows["depth_path"].tolist(),
                         self.rows["mask_path"].tolist()))

        def _work(item: tuple[str, str, str]) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
            render_id, depth_path, mask_path = item
            return render_id, self._pool_target(depth_path, mask_path)

        if worker_count > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for render_id, payload in executor.map(_work, items):
                    self._target_cache[render_id] = payload
        else:
            for item in items:
                render_id, payload = _work(item)
                self._target_cache[render_id] = payload

    def get_target(self, render_id: str) -> tuple[np.ndarray, np.ndarray]:
        if render_id in self._target_cache:
            return self._target_cache[render_id]
        row = self.rows[self.rows["render_id"] == str(render_id)].iloc[0]
        target = self._pool_target(row["depth_path"], row["mask_path"])
        if self._cache_targets:
            self._target_cache[str(render_id)] = target
        return target

    def __getitem__(self, index: int) -> DenseDepthSample:
        row = self.rows.iloc[int(index)]
        cache_index = int(row["cache_index"])
        features = np.asarray(self._patch_features[cache_index], dtype=np.float32)
        if self.feature_mode == "patch_cls":
            assert self._cls_features is not None
            cls = np.asarray(self._cls_features[cache_index], dtype=np.float32)
            cls_grid = np.broadcast_to(
                cls[None, None, :],
                (*features.shape[:2], cls.shape[0]),
            )
            features = np.concatenate([features, cls_grid], axis=-1)
        target, valid = self.get_target(str(row["render_id"]))
        return DenseDepthSample(
            render_id=str(row["render_id"]),
            features=features,
            target=target,
            valid=valid,
        )

    def materialize_arrays(self) -> dict[str, Any]:
        """Stack the dataset into dense numpy arrays for training."""
        features_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        valid_list: list[np.ndarray] = []
        kept: list[int] = []
        for index in range(len(self)):
            sample = self[index]
            if int(sample.valid.sum()) < self._min_valid_patches:
                continue
            features_list.append(sample.features)
            targets_list.append(sample.target)
            valid_list.append(sample.valid)
            kept.append(index)
        if not features_list:
            return {
                "features": np.zeros((0, *self.patch_grid_shape, self.feature_dim), dtype=np.float32),
                "targets": np.zeros((0, *self.patch_grid_shape), dtype=np.float32),
                "valid": np.zeros((0, *self.patch_grid_shape), dtype=bool),
                "render_ids": np.asarray([], dtype=str),
                "rows": self.rows.iloc[:0].reset_index(drop=True),
            }
        return {
            "features": np.stack(features_list, axis=0).astype(np.float32),
            "targets": np.stack(targets_list, axis=0).astype(np.float32),
            "valid": np.stack(valid_list, axis=0).astype(bool),
            "render_ids": np.asarray(
                [self.rows.iloc[i]["render_id"] for i in kept], dtype=str
            ),
            "rows": self.rows.iloc[kept].reset_index(drop=True),
        }

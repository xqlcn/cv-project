"""Storage contract for Experiment 1 feature caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np


class FeatureCacheError(ValueError):
    """Raised when a feature cache violates the Experiment 1 contract."""


def _as_render_ids(render_ids: Sequence[Any]) -> np.ndarray:
    ids = np.asarray([str(render_id) for render_id in render_ids])
    if ids.ndim != 1:
        raise FeatureCacheError("render_ids must be one-dimensional")
    if len(set(ids.tolist())) != len(ids):
        raise FeatureCacheError("render_ids must be unique")
    if any(str(render_id) == "" for render_id in ids):
        raise FeatureCacheError("render_ids must not contain empty strings")
    return ids


def _as_features(features: Any, *, expected_rows: int) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise FeatureCacheError(f"features must be [N,D], got shape {arr.shape}")
    if arr.shape[0] != int(expected_rows):
        raise FeatureCacheError(
            f"features row count {arr.shape[0]} does not match render_ids "
            f"count {expected_rows}"
        )
    if not np.isfinite(arr).all():
        raise FeatureCacheError("features contain NaN or Inf values")
    return arr


def validate_feature_cache(
    render_ids: Sequence[Any],
    features: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize feature-cache arrays."""
    ids = _as_render_ids(render_ids)
    feats = _as_features(features, expected_rows=len(ids))
    return ids, feats


def save_feature_cache(
    path: Union[str, Path],
    *,
    render_ids: Sequence[Any],
    features: Any,
    metadata: Optional[Mapping[str, Any]] = None,
    compressed: bool = True,
) -> Path:
    """Save an NPZ cache containing render_ids and features arrays."""
    ids, feats = validate_feature_cache(render_ids, features)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "render_ids": ids.astype(str),
        "features": feats.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(dict(metadata or {}), sort_keys=True)),
    }
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path


def load_feature_cache(path: Union[str, Path]) -> dict[str, Any]:
    """Load and validate an Experiment 1 feature cache."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "render_ids" not in data or "features" not in data:
            raise FeatureCacheError("Feature cache must contain render_ids/features")
        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            raw = str(np.asarray(data["metadata_json"]).item())
            metadata = json.loads(raw) if raw else {}
        ids, feats = validate_feature_cache(data["render_ids"], data["features"])
    return {"render_ids": ids, "features": feats, "metadata": metadata}


def feature_cache_path(
    feature_dir: Union[str, Path],
    *,
    model_name: str,
    layer_name: str,
) -> Path:
    """Return the canonical cache path for a model/layer pair."""
    return Path(feature_dir) / str(model_name) / f"{str(layer_name)}.npz"


def _as_patch_features(features: Any, *, expected_rows: int) -> np.ndarray:
    arr = np.asarray(features)
    if arr.ndim != 4:
        raise FeatureCacheError(
            f"patch features must be [N,P,P,D], got shape {arr.shape}"
        )
    if arr.shape[0] != int(expected_rows):
        raise FeatureCacheError(
            f"patch features row count {arr.shape[0]} does not match render_ids "
            f"count {expected_rows}"
        )
    if arr.shape[1] != arr.shape[2]:
        raise FeatureCacheError(
            f"patch grid must be square, got {arr.shape[1]}x{arr.shape[2]}"
        )
    if not np.isfinite(arr).all():
        raise FeatureCacheError("patch features contain NaN or Inf values")
    return arr


def _as_cls_features(features: Any, *, expected_rows: int) -> np.ndarray:
    arr = np.asarray(features)
    if arr.ndim != 2:
        raise FeatureCacheError(f"cls features must be [N,D], got shape {arr.shape}")
    if arr.shape[0] != int(expected_rows):
        raise FeatureCacheError(
            f"cls features row count {arr.shape[0]} does not match render_ids "
            f"count {expected_rows}"
        )
    if not np.isfinite(arr).all():
        raise FeatureCacheError("cls features contain NaN or Inf values")
    return arr


def save_patch_feature_cache(
    path: Union[str, Path],
    *,
    render_ids: Sequence[Any],
    patch_features: Any,
    cls_features: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    dtype: str = "float16",
    compressed: bool = False,
) -> Path:
    """Save a 4D (N,P,P,D) patch feature cache as NPZ.

    Defaults to float16 storage to keep cache size tractable. We avoid
    ``savez_compressed`` for these large arrays because the gain is small for
    already low-precision floats while compression cost is significant.
    ``cls_features`` is optional and keeps the cache backward-compatible with
    older patch-only files.
    """
    ids = _as_render_ids(render_ids)
    arr = _as_patch_features(patch_features, expected_rows=len(ids))
    target_dtype = np.dtype(dtype)
    arr = arr.astype(target_dtype, copy=False)
    cls_arr = None
    if cls_features is not None:
        cls_arr = _as_cls_features(cls_features, expected_rows=len(ids)).astype(
            target_dtype,
            copy=False,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "render_ids": ids.astype(str),
        "patch_features": arr,
        "patch_grid_shape": np.asarray(arr.shape[1:3], dtype=np.int32),
        "feature_dim": np.asarray(arr.shape[3], dtype=np.int32),
        "metadata_json": np.asarray(
            json.dumps(dict(metadata or {}), sort_keys=True)
        ),
    }
    if cls_arr is not None:
        payload["cls_features"] = cls_arr
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path


def load_patch_feature_cache(
    path: Union[str, Path],
    *,
    mmap_mode: Optional[str] = "r",
) -> dict[str, Any]:
    """Load a 4D patch feature cache.

    When ``mmap_mode`` is set we use ``np.load`` with ``mmap_mode`` so the
    large patch tensor stays on disk; this is the default for training.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    try:
        if "render_ids" not in data or "patch_features" not in data:
            raise FeatureCacheError(
                "Patch feature cache must contain render_ids/patch_features"
            )
        ids = _as_render_ids(data["render_ids"])
        patch = data["patch_features"]
        if patch.ndim != 4:
            raise FeatureCacheError(
                f"patch features must be [N,P,P,D], got shape {patch.shape}"
            )
        if patch.shape[0] != len(ids):
            raise FeatureCacheError(
                f"patch features row count {patch.shape[0]} does not match "
                f"render_ids count {len(ids)}"
            )
        cls_features = None
        if "cls_features" in data.files:
            cls_features = data["cls_features"]
            if cls_features.ndim != 2:
                raise FeatureCacheError(
                    f"cls features must be [N,D], got shape {cls_features.shape}"
                )
            if cls_features.shape[0] != len(ids):
                raise FeatureCacheError(
                    f"cls features row count {cls_features.shape[0]} does not "
                    f"match render_ids count {len(ids)}"
                )
        metadata: dict[str, Any] = {}
        if "metadata_json" in data.files:
            raw = str(np.asarray(data["metadata_json"]).item())
            metadata = json.loads(raw) if raw else {}
        grid_shape = (
            tuple(int(v) for v in np.asarray(data["patch_grid_shape"]).tolist())
            if "patch_grid_shape" in data.files
            else (int(patch.shape[1]), int(patch.shape[2]))
        )
        feature_dim = (
            int(np.asarray(data["feature_dim"]).item())
            if "feature_dim" in data.files
            else int(patch.shape[3])
        )
    except Exception:
        data.close()
        raise
    return {
        "render_ids": ids,
        "patch_features": patch,
        "cls_features": cls_features,
        "patch_grid_shape": grid_shape,
        "feature_dim": feature_dim,
        "metadata": metadata,
        "_npz_handle": data,
    }


def patch_feature_cache_path(
    feature_dir: Union[str, Path],
    *,
    model_name: str,
    layer_name: str,
) -> Path:
    """Return the canonical patch cache path for a model/layer pair."""
    return Path(feature_dir) / str(model_name) / f"{str(layer_name)}_patch.npz"

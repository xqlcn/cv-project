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

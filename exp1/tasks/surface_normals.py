"""Aggregate surface-normal labels for Experiment 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd


def _resolve_path(project_root: Optional[Union[str, Path]], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


def aggregate_surface_normal_label(
    normal: np.ndarray,
    mask: np.ndarray,
    *,
    min_valid_pixels: int = 64,
) -> Dict[str, Any]:
    """Compute one normalized foreground mean normal label."""
    mask_bool = mask.astype(bool)
    valid = mask_bool & np.isfinite(normal).all(axis=-1)
    valid_count = int(valid.sum())
    if valid_count < int(min_valid_pixels):
        return {
            "mean_normal_x": np.nan,
            "mean_normal_y": np.nan,
            "mean_normal_z": np.nan,
            "normal_valid_pixel_count": valid_count,
            "label_valid": False,
            "label_error_message": "insufficient_valid_normal_pixels",
        }

    mean = normal[valid].astype(np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 1e-8:
        return {
            "mean_normal_x": np.nan,
            "mean_normal_y": np.nan,
            "mean_normal_z": np.nan,
            "normal_valid_pixel_count": valid_count,
            "label_valid": False,
            "label_error_message": "degenerate_mean_normal",
        }

    mean = mean / norm
    return {
        "mean_normal_x": float(mean[0]),
        "mean_normal_y": float(mean[1]),
        "mean_normal_z": float(mean[2]),
        "normal_valid_pixel_count": valid_count,
        "label_valid": True,
        "label_error_message": "",
    }


def build_surface_normal_aggregate_labels(
    rows: Iterable[Mapping[str, Any]],
    *,
    project_root: Optional[Union[str, Path]] = None,
    min_valid_pixels: int = 64,
) -> pd.DataFrame:
    """Build aggregate normal labels for render manifest rows."""
    labels = []
    for row in rows:
        out: Dict[str, Any] = {"render_id": str(row["render_id"])}
        try:
            normal = np.load(
                _resolve_path(project_root, row["normal_path"]),
                allow_pickle=False,
            )
            mask = np.load(
                _resolve_path(project_root, row["mask_path"]),
                allow_pickle=False,
            )
            out.update(
                aggregate_surface_normal_label(
                    normal,
                    mask,
                    min_valid_pixels=min_valid_pixels,
                )
            )
        except Exception as exc:
            out.update(
                {
                    "mean_normal_x": np.nan,
                    "mean_normal_y": np.nan,
                    "mean_normal_z": np.nan,
                    "normal_valid_pixel_count": 0,
                    "label_valid": False,
                    "label_error_message": f"{type(exc).__name__}:{exc}",
                }
            )
        labels.append(out)
    return pd.DataFrame(labels)

"""Camera and viewpoint labels derived from render metadata."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import pandas as pd


def _angle_features(deg: Any) -> tuple[float, float]:
    rad = math.radians(float(deg))
    return float(math.sin(rad)), float(math.cos(rad))


def build_camera_labels(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build camera-distance and viewpoint labels keyed by render_id."""
    labels = []
    for row in rows:
        distance = float(row["camera_distance"])
        az_sin, az_cos = _angle_features(row["camera_azimuth_deg"])
        el_sin, el_cos = _angle_features(row["camera_elevation_deg"])
        labels.append(
            {
                "render_id": str(row["render_id"]),
                "camera_distance": distance,
                "log_camera_distance": math.log(max(distance, 1e-8)),
                "camera_azimuth_deg": float(row["camera_azimuth_deg"]),
                "camera_elevation_deg": float(row["camera_elevation_deg"]),
                "camera_fov_deg": float(row["camera_fov_deg"]),
                "azimuth_sin": az_sin,
                "azimuth_cos": az_cos,
                "elevation_sin": el_sin,
                "elevation_cos": el_cos,
                "label_valid": True,
                "label_error_message": "",
            }
        )
    return pd.DataFrame(labels)

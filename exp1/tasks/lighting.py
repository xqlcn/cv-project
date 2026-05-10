"""Lighting labels derived from render metadata."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import pandas as pd


def light_direction_from_angles(
    azimuth_deg: Any,
    elevation_deg: Any,
) -> tuple[float, float, float]:
    """Convert azimuth/elevation degrees to a unit direction vector."""
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    return (
        float(math.cos(el) * math.cos(az)),
        float(math.cos(el) * math.sin(az)),
        float(math.sin(el)),
    )


def build_lighting_labels(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build lighting direction/intensity labels keyed by render_id."""
    labels = []
    for row in rows:
        intensity = float(row["light_intensity"])
        x, y, z = light_direction_from_angles(
            row["light_azimuth_deg"],
            row["light_elevation_deg"],
        )
        labels.append(
            {
                "render_id": str(row["render_id"]),
                "light_type": str(row.get("light_type", "")),
                "light_azimuth_deg": float(row["light_azimuth_deg"]),
                "light_elevation_deg": float(row["light_elevation_deg"]),
                "light_intensity": intensity,
                "log_light_intensity": math.log(max(intensity, 1e-8)),
                "light_dir_x": x,
                "light_dir_y": y,
                "light_dir_z": z,
                "label_valid": True,
                "label_error_message": "",
            }
        )
    return pd.DataFrame(labels)

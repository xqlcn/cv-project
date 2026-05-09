"""Blender lighting setup for Experiment 1 render rows."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def setup_lighting(
    record: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> Dict[str, float]:
    """Create deterministic scene lighting from one render-plan row."""
    from lighting_utils import add_area_fill, add_sun, clear_lights

    clear_lights()
    light_type = str(record.get("light_type", "sun")).lower()
    if light_type != "sun":
        raise ValueError(f"Unsupported light_type for Blender skeleton: {light_type}")

    elevation = float(record["light_elevation_deg"])
    azimuth = float(record["light_azimuth_deg"])
    intensity = float(record["light_intensity"])
    add_sun(
        "Exp1KeySun",
        elevation_deg=elevation,
        azimuth_deg=azimuth,
        energy=intensity,
    )

    lighting_cfg = cfg.get("lighting_grid", {})
    fill_intensity = float(lighting_cfg.get("fill_intensity", 0.35))
    if fill_intensity > 0:
        add_area_fill(
            "Exp1FillArea",
            location=(2.0, -2.5, 3.0),
            energy=fill_intensity,
        )

    return {
        "light_azimuth_deg": azimuth,
        "light_elevation_deg": elevation,
        "light_intensity": intensity,
        "fill_intensity": fill_intensity,
    }

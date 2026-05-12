"""Blender camera setup for Experiment 1 render rows."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping


def setup_camera(record: Mapping[str, Any], cfg: Mapping[str, Any]):
    """Create and place the Blender camera from one render-plan row."""
    from camera_utils import ensure_camera, set_camera_extrinsics, set_camera_intrinsics

    render_cfg = cfg.get("render", {})
    camera_cfg = render_cfg.get("camera", {})

    fov_deg = float(record.get("camera_fov_deg", camera_cfg.get("fov_deg", 50.0)))
    clip_start = float(camera_cfg.get("clip_start", 0.01))
    clip_end = float(camera_cfg.get("clip_end", 1000.0))

    cam_obj = ensure_camera()
    set_camera_intrinsics(
        cam_obj.data,
        fov_deg=fov_deg,
        clip_start=clip_start,
        clip_end=clip_end,
    )
    set_camera_extrinsics(
        cam_obj,
        azimuth=math.radians(float(record["camera_azimuth_deg"])),
        elevation=math.radians(float(record["camera_elevation_deg"])),
        distance=float(record["camera_distance"]),
    )
    return cam_obj


def camera_metadata(cam_obj) -> Dict[str, float]:
    """Return camera pose fields that are useful for render metadata."""
    quat = cam_obj.rotation_euler.to_quaternion()
    return {
        "camera_location_x": float(cam_obj.location.x),
        "camera_location_y": float(cam_obj.location.y),
        "camera_location_z": float(cam_obj.location.z),
        "camera_rotation_quat_w": float(quat.w),
        "camera_rotation_quat_x": float(quat.x),
        "camera_rotation_quat_y": float(quat.y),
        "camera_rotation_quat_z": float(quat.z),
    }

"""Camera setup helpers for controlled viewpoints (Blender)."""

from __future__ import annotations

import math
from typing import Tuple

import bpy
import mathutils


def ensure_camera(name: str = "ProbeCamera") -> bpy.types.Object:
    cam_data = bpy.data.cameras.get(name) or bpy.data.cameras.new(name)
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj
    return obj


def set_camera_extrinsics(
    cam_obj: bpy.types.Object,
    *,
    azimuth: float,
    elevation: float,
    distance: float,
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """
    Place camera on a sphere around target using right-handed spherical coords.

    azimuth: radians, rotation in XY plane from +X toward +Y (Blender convention).
    elevation: radians, angle above XY plane (Z component increases with elevation).
    """
    t = mathutils.Vector(target)
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)
    cam_obj.location = t + mathutils.Vector((x, y, z))

    direction = t - cam_obj.location
    direction.normalize()
    quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = quat.to_euler()


def set_camera_intrinsics(
    cam_data: bpy.types.Camera,
    *,
    fov_deg: float,
    clip_start: float = 0.01,
    clip_end: float = 1000.0,
) -> None:
    cam_data.type = "PERSP"
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(fov_deg)
    cam_data.clip_start = clip_start
    cam_data.clip_end = clip_end

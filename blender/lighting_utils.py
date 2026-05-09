"""Lighting setup for reproducible renders (Blender)."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import bpy
import mathutils


def clear_lights() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)


def add_sun(
    name: str,
    *,
    elevation_deg: float,
    azimuth_deg: float,
    energy: float,
) -> bpy.types.Object:
    """
    Directional sun: angles describe where the light points FROM (approximate sun disk).
    """
    light_data = bpy.data.lights.new(name=name, type="SUN")
    light_data.energy = energy
    obj = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(obj)

    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    direction = mathutils.Vector(
        (
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        )
    )
    quat = (-direction).to_track_quat("-Z", "Y")
    obj.rotation_euler = quat.to_euler()
    return obj


def add_area_fill(
    name: str,
    *,
    location: Tuple[float, float, float],
    energy: float,
    size: float = 4.0,
) -> bpy.types.Object:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.size = size
    obj = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    return obj

"""Geometry-buffer extraction for the Blender render skeleton."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _camera_frame_world(scene: Any, cam_obj: Any):
    frame = cam_obj.data.view_frame(scene=scene)
    matrix = cam_obj.matrix_world
    return [matrix @ corner for corner in frame]


def _frame_corners(scene: Any, cam_obj: Any):
    # Blender returns frame corners in camera-local coordinates. The order is
    # bottom-left, bottom-right, top-right, top-left for current Blender builds.
    world = _camera_frame_world(scene, cam_obj)
    bottom_left, bottom_right, top_right, top_left = world
    return bottom_left, bottom_right, top_right, top_left


def _lerp(a: Any, b: Any, t: float):
    return a + (b - a) * t


def _evaluated_depsgraph(scene: Any):
    if hasattr(scene, "evaluated_depsgraph_get"):
        return scene.evaluated_depsgraph_get()

    import bpy  # type: ignore[reportMissingImports]

    return bpy.context.evaluated_depsgraph_get()


def render_geometry_buffers(
    *,
    scene: Any,
    cam_obj: Any,
    width: int,
    height: int,
) -> Dict[str, np.ndarray]:
    """Ray-cast depth, camera-space normal, and foreground mask arrays."""
    depsgraph = _evaluated_depsgraph(scene)
    origin = cam_obj.matrix_world.translation.copy()
    cam_inv = cam_obj.matrix_world.inverted()
    rot_world_to_camera = cam_inv.to_3x3()
    bottom_left, bottom_right, top_right, top_left = _frame_corners(scene, cam_obj)

    depth = np.full((height, width), np.nan, dtype=np.float32)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.bool_)

    clip_end = float(cam_obj.data.clip_end)
    for y in range(height):
        v = 1.0 - ((y + 0.5) / float(height))
        left = _lerp(bottom_left, top_left, v)
        right = _lerp(bottom_right, top_right, v)
        for x in range(width):
            u = (x + 0.5) / float(width)
            target = _lerp(left, right, u)
            direction = target - origin
            direction.normalize()
            hit, location, normal_world, _, _, _ = scene.ray_cast(
                depsgraph,
                origin,
                direction,
                distance=clip_end,
            )
            if not hit:
                continue
            local_hit = cam_inv @ location
            depth[y, x] = np.float32(max(0.0, -float(local_hit.z)))
            n_cam = rot_world_to_camera @ normal_world
            n_len = max(float(n_cam.length), 1e-8)
            normal[y, x, :] = np.asarray(
                (n_cam.x / n_len, n_cam.y / n_len, n_cam.z / n_len),
                dtype=np.float32,
            )
            mask[y, x] = True

    return {"depth": depth, "normal_camera": normal, "mask": mask}


def buffer_shapes(buffers: Dict[str, np.ndarray]) -> Dict[str, Tuple[int, ...]]:
    return {name: tuple(array.shape) for name, array in buffers.items()}

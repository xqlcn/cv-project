"""
Mesh import, normalization, and mirroring for Blender renders.

Runs inside Blender's Python environment (bpy available).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import bpy
import mathutils


def _parse_off_file(filepath: str) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, ...]]]:
    """
    Parse ASCII OFF (ModelNet). Handles header 'OFF' on its own line or merged with counts.

    Returns (vertices, faces) where faces are tuples of vertex indices (triangles or quads).
    """
    tokens: List[str] = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tokens.extend(line.split())

    if not tokens:
        raise ValueError(f"Empty OFF file: {filepath}")

    idx = 0
    if tokens[idx].upper() == "OFF":
        idx += 1

    try:
        num_v = int(tokens[idx])
        num_f = int(tokens[idx + 1])
        # third number is edge count (often 0); skip
        idx += 3
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Bad OFF header in {filepath}") from exc

    verts: List[Tuple[float, float, float]] = []
    for _ in range(num_v):
        try:
            verts.append(
                (
                    float(tokens[idx]),
                    float(tokens[idx + 1]),
                    float(tokens[idx + 2]),
                )
            )
            idx += 3
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Bad OFF vertices in {filepath}") from exc

    faces: List[Tuple[int, ...]] = []
    for _ in range(num_f):
        try:
            n = int(tokens[idx])
            idx += 1
            if n < 3:
                raise ValueError(f"Face degree {n} < 3")
            face_idxs = tuple(int(tokens[idx + j]) for j in range(n))
            idx += n
            faces.append(face_idxs)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Bad OFF faces in {filepath}") from exc

    return verts, faces


def import_off(filepath: str, *, object_name: str = "ImportedOFF") -> bpy.types.Object:
    """Create a mesh object from an ASCII .off file (no Blender addon required)."""
    verts, faces = _parse_off_file(filepath)
    mesh = bpy.data.meshes.new(object_name + "_mesh")
    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    return obj


def clear_meshes(objects: Optional[Iterable[bpy.types.Object]] = None) -> None:
    """Remove mesh objects from the scene (default: all MESH objects)."""
    if objects is None:
        objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.delete()


def import_mesh(filepath: str) -> bpy.types.Object:
    """
    Import a mesh file. Supports .off (ModelNet), .obj, .glb/.gltf, .fbx.

    Returns the primary imported mesh object (first mesh if multiple).
    """
    path_lower = filepath.lower()
    bpy.ops.object.select_all(action="DESELECT")

    if path_lower.endswith(".off"):
        return import_off(filepath, object_name="MeshOFF")

    if path_lower.endswith(".obj"):
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=filepath)
        else:
            bpy.ops.import_scene.obj(filepath=filepath)
    elif path_lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif path_lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=filepath)
    else:
        raise ValueError(f"Unsupported mesh extension: {filepath}")

    imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not imported:
        raise RuntimeError(f"No mesh imported from {filepath}")
    return imported[0]


def _bbox_corners_world(obj: bpy.types.Object) -> List[mathutils.Vector]:
    world = obj.matrix_world
    local_bb = [mathutils.Vector(corner) for corner in obj.bound_box]
    return [world @ corner for corner in local_bb]


def center_and_normalize(
    obj: bpy.types.Object,
    target_size: float = 1.0,
) -> Tuple[mathutils.Vector, float]:
    """
    Center mesh at origin and uniform-scale so max axis-aligned extent equals target_size.

    Returns (previous_world_center, applied_uniform_scale).
    """
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    corners = _bbox_corners_world(obj)
    min_c = mathutils.Vector(
        (min(v[i] for v in corners) for i in range(3))
    )
    max_c = mathutils.Vector(
        (max(v[i] for v in corners) for i in range(3))
    )
    center = (min_c + max_c) / 2.0
    size_vec = max_c - min_c
    max_extent = max(size_vec.x, size_vec.y, size_vec.z)
    if max_extent <= 1e-8:
        scale = 1.0
    else:
        scale = target_size / max_extent

    obj.matrix_world.translation -= center
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
    return center, scale


def duplicate_mesh_mirror(
    obj: bpy.types.Object,
    axis: str = "X",
) -> bpy.types.Object:
    """
    Duplicate mesh geometry and mirror across the given axis in object space (X/Y/Z).

    The duplicate is a separate object with its own mesh data.
    """
    axis = axis.upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("axis must be X, Y, or Z")

    dup = obj.copy()
    dup.data = obj.data.copy()
    dup.animation_data_clear()
    coll = obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection
    coll.objects.link(dup)
    dup.matrix_world = obj.matrix_world.copy()

    if axis == "X":
        dup.scale.x *= -1.0
    elif axis == "Y":
        dup.scale.y *= -1.0
    else:
        dup.scale.z *= -1.0

    bpy.ops.object.select_all(action="DESELECT")
    dup.select_set(True)
    bpy.context.view_layer.objects.active = dup
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode="OBJECT")

    return dup


def delete_object(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.delete()

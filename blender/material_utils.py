"""Material assignment for texture-condition experiments (Blender)."""

from __future__ import annotations

import math
import random
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import bpy
except ImportError:
    bpy = None


GENERATED_MATERIAL_SUFFIXES = (
    "_flat_mat",
    "_noise_mat",
    "_photorealistic_fallback_mat",
)


def _material_slots(obj: bpy.types.Object) -> Iterable[Optional[bpy.types.Material]]:
    if not hasattr(obj.data, "materials"):
        return []
    return list(obj.data.materials)


def _has_image_texture(mat: bpy.types.Material) -> bool:
    if not mat.use_nodes or mat.node_tree is None:
        return False
    for node in mat.node_tree.nodes:
        if node.bl_idname == "ShaderNodeTexImage" and getattr(node, "image", None):
            return True
    return False


def _is_generated_material(mat: bpy.types.Material) -> bool:
    name = mat.name.lower()
    return any(name.endswith(suffix) for suffix in GENERATED_MATERIAL_SUFFIXES)


def has_preservable_material(obj: bpy.types.Object) -> bool:
    """Return whether an imported mesh appears to carry a real material."""
    for mat in _material_slots(obj):
        if mat is None or _is_generated_material(mat):
            continue
        if _has_image_texture(mat):
            return True
        if mat.use_nodes and mat.node_tree is not None and len(mat.node_tree.nodes) > 0:
            return True
        diffuse = tuple(float(v) for v in getattr(mat, "diffuse_color", (1, 1, 1, 1)))
        if diffuse[:3] != (1.0, 1.0, 1.0):
            return True
    return False


def _replace_all_material_slots(
    obj: bpy.types.Object,
    mat: bpy.types.Material,
) -> None:
    if not obj.data.materials:
        obj.data.materials.append(mat)
        return
    for idx in range(len(obj.data.materials)):
        obj.data.materials[idx] = mat


def _new_principled_material(
    name: str,
) -> tuple[bpy.types.Material, bpy.types.Node, Any, Any]:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    links.new(principled.outputs["BSDF"], out.inputs["Surface"])
    return mat, principled, nodes, links


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _random_rgba_bytes(*, seed: int, width: int, height: int) -> bytes:
    rng = random.Random(int(seed))
    pixels = bytearray()
    for _ in range(int(width) * int(height)):
        pixels.extend((rng.randrange(256), rng.randrange(256), rng.randrange(256), 255))
    return bytes(pixels)


def _write_rgba_png(
    path: Path,
    *,
    width: int,
    height: int,
    pixels: bytes,
) -> None:
    row_stride = int(width) * 4
    raw_rows = bytearray()
    for row_idx in range(int(height)):
        raw_rows.append(0)  # PNG filter type 0: no filtering.
        start = row_idx * row_stride
        raw_rows.extend(pixels[start : start + row_stride])

    png = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", int(width), int(height), 8, 6, 0, 0, 0),
            ),
            _png_chunk(b"IDAT", zlib.compress(bytes(raw_rows))),
            _png_chunk(b"IEND", b""),
        )
    )
    path.write_bytes(png)


def _create_random_image_texture(
    path: Path,
    *,
    seed: int,
    image_size: tuple = (256, 256),
    color_space: str = "sRGB",
) -> bpy.types.Image:
    width, height = [max(1, int(v)) for v in image_size]
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = _random_rgba_bytes(seed=int(seed), width=width, height=height)
    _write_rgba_png(path, width=width, height=height, pixels=pixels)

    image = bpy.data.images.load(str(path), check_existing=False)
    image.name = f"exp1_random_texture_{int(seed)}"
    try:
        image.colorspace_settings.name = str(color_space)
    except Exception:
        pass
    return image


def _set_input(node: bpy.types.Node, name: str, value) -> None:
    if name in node.inputs:
        node.inputs[name].default_value = value


def _build_procedural_noise_shader(
    *,
    nodes: Any,
    links: Any,
    principled: bpy.types.Node,
    seed: int,
    scale_range: tuple,
    detail: float,
    mapping_mode: str,
) -> None:
    """Build a colorful, high-contrast procedural RGB noise shader.

    Uses three independent noise textures (one per RGB channel), each routed
    through a Mapping node with a unique per-seed offset/rotation, then
    contrast-expanded so the channel covers most of [0, 1]. This avoids the
    "flat muddy color" pathology that occurs when a single noise Fac is
    routed through a 2-stop random-color ramp (the two stops often end up
    in the same color region, producing a near-uniform surface).
    """
    rng = random.Random(int(seed))
    low, high = [float(v) for v in scale_range]
    if high < low:
        low, high = high, low
    base_scale = low + rng.random() * max(high - low, 0.0)
    if base_scale <= 0.0:
        base_scale = max(low, 1.0)

    coord = nodes.new(type="ShaderNodeTexCoord")
    mode = str(mapping_mode).strip().lower()
    if mode == "uv":
        vector_output = "UV"
    elif mode == "generated":
        vector_output = "Generated"
    else:
        vector_output = "Object"

    channel_values = []
    for _ in range(3):
        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.inputs["Location"].default_value = (
            rng.uniform(-200.0, 200.0),
            rng.uniform(-200.0, 200.0),
            rng.uniform(-200.0, 200.0),
        )
        mapping.inputs["Rotation"].default_value = (
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-math.pi, math.pi),
        )
        links.new(coord.outputs[vector_output], mapping.inputs["Vector"])

        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = base_scale * (0.7 + rng.random() * 0.6)
        noise.inputs["Detail"].default_value = float(detail)
        if "Roughness" in noise.inputs:
            noise.inputs["Roughness"].default_value = 0.45 + rng.random() * 0.45
        if "Distortion" in noise.inputs:
            noise.inputs["Distortion"].default_value = 0.4 + rng.random() * 1.4
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        # Contrast-expand the Fac (centred around ~0.5 for Perlin noise) so the
        # channel actually spans most of [0, 1].
        sub = nodes.new(type="ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = 0.5
        links.new(noise.outputs["Fac"], sub.inputs[0])

        mul = nodes.new(type="ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 3.0
        links.new(sub.outputs["Value"], mul.inputs[0])

        add = nodes.new(type="ShaderNodeMath")
        add.operation = "ADD"
        add.inputs[1].default_value = 0.5
        add.use_clamp = True
        links.new(mul.outputs["Value"], add.inputs[0])

        channel_values.append(add)

    combine = nodes.new(type="ShaderNodeCombineXYZ")
    links.new(channel_values[0].outputs["Value"], combine.inputs["X"])
    links.new(channel_values[1].outputs["Value"], combine.inputs["Y"])
    links.new(channel_values[2].outputs["Value"], combine.inputs["Z"])
    links.new(combine.outputs["Vector"], principled.inputs["Base Color"])


def assign_principled_material(
    obj: bpy.types.Object,
    *,
    texture_type: str,
    base_color: tuple = (0.72, 0.65, 0.55, 1.0),
    seed: int = 0,
    roughness: float = 0.55,
    noise_scale_range: tuple = (8.0, 20.0),
    noise_detail: float = 6.0,
    noise_mapping: str = "object",
    image_texture_path: Optional[str] = None,
    image_size: tuple = (256, 256),
    image_color_space: str = "sRGB",
    preserve_existing: bool = True,
) -> bpy.types.Material:
    """
    texture_type: photorealistic | flat | noise

    photorealistic: keep existing material slots if present; otherwise flat fallback.
    """
    texture_type = texture_type.lower()
    if texture_type == "random_noise":
        texture_type = "noise"
    if texture_type == "image_noise":
        texture_type = "noise"

    existing_material = next((mat for mat in _material_slots(obj) if mat), None)
    if (
        texture_type == "photorealistic"
        and preserve_existing
        and existing_material is not None
    ):
        return existing_material

    suffix = (
        "photorealistic_fallback" if texture_type == "photorealistic" else texture_type
    )
    mat_name = f"{obj.name}_{suffix}_mat"
    mat, principled, nodes, links = _new_principled_material(mat_name)

    if texture_type in {"photorealistic", "flat"}:
        _set_input(principled, "Base Color", base_color)
        _set_input(principled, "Roughness", float(roughness))
    elif texture_type == "noise":
        if image_texture_path:
            image = _create_random_image_texture(
                Path(image_texture_path),
                seed=seed,
                image_size=image_size,
                color_space=image_color_space,
            )
            tex = nodes.new(type="ShaderNodeTexImage")
            tex.image = image
            tex.extension = "REPEAT"
            coord = nodes.new(type="ShaderNodeTexCoord")
            mapping_mode = str(noise_mapping).strip().lower()
            if mapping_mode == "uv":
                vector_output = (
                    "UV" if getattr(obj.data, "uv_layers", None) else "Generated"
                )
            elif mapping_mode == "generated":
                vector_output = "Generated"
            else:
                vector_output = "Object"
            links.new(coord.outputs[vector_output], tex.inputs["Vector"])
            links.new(tex.outputs["Color"], principled.inputs["Base Color"])
        else:
            _build_procedural_noise_shader(
                nodes=nodes,
                links=links,
                principled=principled,
                seed=int(seed),
                scale_range=tuple(noise_scale_range),
                detail=float(noise_detail),
                mapping_mode=str(noise_mapping),
            )
        _set_input(principled, "Roughness", float(roughness))
    else:
        raise ValueError(f"Unknown texture_type: {texture_type}")

    _replace_all_material_slots(obj, mat)
    return mat

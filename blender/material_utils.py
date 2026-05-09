"""Material assignment for texture-condition experiments (Blender)."""

from __future__ import annotations

import random
from typing import Optional

import bpy


def assign_principled_material(
    obj: bpy.types.Object,
    *,
    texture_type: str,
    base_color: tuple = (0.72, 0.65, 0.55, 1.0),
    seed: int = 0,
) -> None:
    """
    texture_type: photorealistic | flat | noise

    photorealistic: keep existing material slots if present; otherwise flat fallback.
    """
    texture_type = texture_type.lower()
    if not obj.data.materials:
        obj.data.materials.append(None)

    mat_name = f"{obj.name}_{texture_type}_mat"
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    links.new(principled.outputs["BSDF"], out.inputs["Surface"])

    if texture_type == "photorealistic":
        # If object already has non-empty materials from import, skip override on slot 0
        existing = obj.data.materials[0] if obj.data.materials else None
        if existing and existing.name != mat_name and len(existing.node_tree.nodes) > 2:
            return
        principled.inputs["Base Color"].default_value = base_color
        principled.inputs["Roughness"].default_value = 0.45
    elif texture_type == "flat":
        principled.inputs["Base Color"].default_value = base_color
        principled.inputs["Roughness"].default_value = 0.5
    elif texture_type == "noise":
        rng = random.Random(seed)
        tex = nodes.new(type="ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 8.0 + rng.random() * 12.0
        tex.inputs["Detail"].default_value = 6.0
        mapping = nodes.new(type="ShaderNodeMapping")
        coord = nodes.new(type="ShaderNodeTexCoord")
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        mix = nodes.new(type="ShaderNodeMixRGB")
        mix.blend_type = "MIX"
        mix.inputs["Fac"].default_value = 1.0
        mix.inputs["Color1"].default_value = (
            rng.random(),
            rng.random(),
            rng.random(),
            1.0,
        )
        links.new(tex.outputs["Color"], mix.inputs["Color2"])
        links.new(mix.outputs["Color"], principled.inputs["Base Color"])
        principled.inputs["Roughness"].default_value = 0.55
    else:
        raise ValueError(f"Unknown texture_type: {texture_type}")

    obj.data.materials[0] = mat

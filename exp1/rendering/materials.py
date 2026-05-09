"""Material setup for the Blender render skeleton."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _rgba(
    rgb_values: Iterable[Any],
    alpha: float = 1.0,
) -> tuple[float, float, float, float]:
    rgb = [float(v) for v in rgb_values]
    if len(rgb) != 3:
        raise ValueError(f"Expected RGB triplet, got {rgb_values}")
    return (rgb[0], rgb[1], rgb[2], float(alpha))


def material_mode(texture_condition: str) -> str:
    """Map Experiment 1 texture conditions to existing Blender helper modes."""
    condition = texture_condition.lower()
    if condition == "random_noise":
        return "noise"
    return condition


def apply_materials(
    objects: Iterable[Any],
    record: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> None:
    """Apply the render row's material condition to mesh objects."""
    from material_utils import assign_principled_material

    condition = str(record.get("texture_condition", "flat"))
    textures_cfg = cfg.get("textures", {})
    flat_cfg = textures_cfg.get("flat", {})
    photo_cfg = textures_cfg.get("photorealistic", {})

    if condition == "photorealistic":
        color = _rgba(photo_cfg.get("fallback_color_rgb", (0.68, 0.68, 0.68)))
    else:
        color = _rgba(flat_cfg.get("color_rgb", (0.62, 0.62, 0.62)))

    for obj in objects:
        if condition == "photorealistic" and obj.data.materials:
            existing = obj.data.materials[0]
            if existing is not None:
                continue
        assign_principled_material(
            obj,
            texture_type=material_mode(condition),
            base_color=color,
            seed=int(record.get("texture_seed", 0) or 0),
        )

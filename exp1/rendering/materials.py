"""Experiment 1 material policy for Blender render rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from exp1.metadata.schema import VALID_TEXTURE_CONDITIONS


TEXTURELESS_SOURCE_DATASETS = ("synthetic_primitives",)
TEXTURELESS_RAW_EXTENSIONS = (".off",)


def _rgba(
    rgb_values: Iterable[Any],
    alpha: float = 1.0,
) -> tuple[float, float, float, float]:
    rgb = [float(v) for v in rgb_values]
    if len(rgb) != 3:
        raise ValueError(f"Expected RGB triplet, got {rgb_values}")
    return (rgb[0], rgb[1], rgb[2], float(alpha))


def _rgb_list(rgba: Iterable[Any]) -> List[float]:
    return [float(v) for v in list(rgba)[:3]]


def _cfg_list(
    cfg: Mapping[str, Any],
    *keys: str,
    default: Iterable[Any],
) -> List[Any]:
    current: Any = cfg
    for key in keys:
        if not isinstance(current, Mapping):
            return list(default)
        current = current.get(key)
    if current is None:
        return list(default)
    if isinstance(current, (str, bytes)):
        return [current]
    return list(current)


def _truthy(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def photorealistic_fallback_reason(
    record: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> Optional[str]:
    """Return why a photorealistic render should use the documented fallback."""
    available = None
    for key in (
        "has_photorealistic_material",
        "photorealistic_material_available",
        "has_imported_material",
    ):
        if key in record:
            available = _truthy(record.get(key))
            if available is not None:
                break
    if available is False:
        return "manifest_marks_material_unavailable"

    source_dataset = str(record.get("source_dataset", "")).strip().lower()
    textureless_sources = {
        str(source).strip().lower()
        for source in _cfg_list(
            cfg,
            "textures",
            "photorealistic",
            "textureless_source_datasets",
            default=TEXTURELESS_SOURCE_DATASETS,
        )
    }
    if source_dataset and source_dataset in textureless_sources:
        return f"source_dataset_marked_textureless:{source_dataset}"

    raw_path = str(record.get("raw_mesh_path") or record.get("mesh_path") or "")
    if Path(raw_path).suffix.lower() in TEXTURELESS_RAW_EXTENSIONS:
        return f"raw_mesh_extension_textureless:{Path(raw_path).suffix.lower()}"

    return None


def material_mode(texture_condition: str) -> str:
    """Map Experiment 1 texture conditions to existing Blender helper modes."""
    condition = texture_condition.lower()
    if condition == "random_noise":
        return "random_noise"
    return condition


def _random_texture_path(record: Mapping[str, Any]) -> str:
    explicit = record.get("random_texture_path")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    rgb_path = record.get("rgb_path")
    if rgb_path is not None and str(rgb_path).strip():
        return str(Path(str(rgb_path)).with_name("random_texture.png"))
    return ""


def _has_uv_layers(obj: Any) -> bool:
    data = getattr(obj, "data", None)
    uv_layers = getattr(data, "uv_layers", None)
    return bool(uv_layers)


def apply_materials(
    objects: Iterable[Any],
    record: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the render row's material condition and return metadata."""
    from material_utils import assign_principled_material, has_preservable_material

    condition = str(record.get("texture_condition", "flat"))
    if condition not in set(VALID_TEXTURE_CONDITIONS):
        raise ValueError(
            f"Unknown texture_condition {condition!r}; expected one of "
            + ", ".join(VALID_TEXTURE_CONDITIONS)
        )

    mesh_objects = list(objects)
    textures_cfg = cfg.get("textures", {})
    flat_cfg = textures_cfg.get("flat", {})
    photo_cfg = textures_cfg.get("photorealistic", {})
    noise_cfg = textures_cfg.get("random_noise", {})

    flat_color = _rgba(flat_cfg.get("color_rgb", (0.62, 0.62, 0.62)))
    fallback_color = _rgba(photo_cfg.get("fallback_color_rgb", (0.68, 0.68, 0.68)))
    flat_roughness = float(flat_cfg.get("roughness", 0.55))
    noise_roughness = float(noise_cfg.get("roughness", 0.55))
    preserve_imported = bool(photo_cfg.get("preserve_imported_materials", True))
    seed = int(record.get("texture_seed", 0) or 0)

    preserved = 0
    fallback = 0
    overridden = 0
    fallback_reasons: List[str] = []

    if condition == "photorealistic":
        configured_reason = photorealistic_fallback_reason(record, cfg)
        for obj in mesh_objects:
            has_material = bool(has_preservable_material(obj))
            should_preserve = (
                preserve_imported and configured_reason is None and has_material
            )
            if should_preserve:
                preserved += 1
                continue

            reason = configured_reason
            if reason is None and not preserve_imported:
                reason = "preserve_imported_materials_false"
            if reason is None and not has_material:
                reason = "missing_imported_material"
            fallback_reasons.append(str(reason or "photorealistic_fallback"))
            fallback += 1
            assign_principled_material(
                obj,
                texture_type="photorealistic",
                base_color=fallback_color,
                seed=seed,
                roughness=float(photo_cfg.get("fallback_roughness", 0.55)),
                preserve_existing=False,
            )

        if preserved and fallback:
            status = "partial_fallback"
            photo_status = "partial_fallback"
        elif fallback:
            status = "fallback"
            photo_status = "fallback_missing_original"
        else:
            status = "preserved"
            photo_status = "preserved"

        return {
            "material_condition": condition,
            "material_mode": "photorealistic",
            "material_status": status,
            "material_object_count": len(mesh_objects),
            "material_preserved_object_count": preserved,
            "material_fallback_object_count": fallback,
            "material_override_object_count": 0,
            "photorealistic_material_status": photo_status,
            "photorealistic_fallback_reason": ";".join(sorted(set(fallback_reasons))),
            "photorealistic_fallback_color_rgb": _rgb_list(fallback_color),
            "texture_seed_used": seed,
        }

    if condition == "flat":
        for obj in mesh_objects:
            assign_principled_material(
                obj,
                texture_type="flat",
                base_color=flat_color,
                seed=seed,
                roughness=flat_roughness,
                preserve_existing=False,
            )
            overridden += 1
        return {
            "material_condition": condition,
            "material_mode": "flat",
            "material_status": "flat_override",
            "material_object_count": len(mesh_objects),
            "material_preserved_object_count": 0,
            "material_fallback_object_count": 0,
            "material_override_object_count": overridden,
            "photorealistic_material_status": "not_applicable",
            "photorealistic_fallback_reason": "",
            "flat_color_rgb": _rgb_list(flat_color),
            "texture_seed_used": seed,
        }

    noise_scale = noise_cfg.get("scale_range", (8.0, 20.0))
    if not isinstance(noise_scale, (list, tuple)) or len(noise_scale) != 2:
        noise_scale = (8.0, 20.0)
    texture_size = noise_cfg.get("texture_size", (256, 256))
    if not isinstance(texture_size, (list, tuple)) or len(texture_size) != 2:
        texture_size = (256, 256)
    noise_mapping = str(noise_cfg.get("mapping", "object")).strip().lower()
    node_type = str(noise_cfg.get("node_type", "procedural_noise")).strip().lower()
    use_image_texture = node_type in {"image_texture", "image_noise"} or noise_mapping in {
        "uv",
        "uv_image",
    }
    random_texture_path = _random_texture_path(record) if use_image_texture else ""
    if noise_mapping == "uv_image":
        noise_mapping = "uv"
    if noise_mapping not in {"object", "generated", "uv"}:
        noise_mapping = "object"
    for obj in mesh_objects:
        assign_principled_material(
            obj,
            texture_type=material_mode(condition),
            base_color=flat_color,
            seed=seed,
            roughness=noise_roughness,
            noise_scale_range=tuple(float(v) for v in noise_scale),
            noise_detail=float(noise_cfg.get("detail", 6.0)),
            noise_mapping=noise_mapping,
            image_texture_path=random_texture_path,
            image_size=tuple(int(v) for v in texture_size),
            image_color_space=str(noise_cfg.get("color_space", "sRGB")),
            preserve_existing=False,
        )
        overridden += 1
    return {
        "material_condition": condition,
        "material_mode": "procedural_noise",
        "material_status": "random_noise_override",
        "material_object_count": len(mesh_objects),
        "material_preserved_object_count": 0,
        "material_fallback_object_count": 0,
        "material_override_object_count": overridden,
        "photorealistic_material_status": "not_applicable",
        "photorealistic_fallback_reason": "",
        "random_noise_seed": seed,
        "random_noise_texture_type": (
            "image_texture" if use_image_texture else "procedural_noise"
        ),
        "random_noise_node_type": (
            "image_texture" if use_image_texture else "ShaderNodeTexNoise"
        ),
        "random_texture_path": random_texture_path,
        "random_texture_size": [int(v) for v in texture_size],
        "random_texture_mapping": noise_mapping,
        "texture_seed_used": seed,
    }

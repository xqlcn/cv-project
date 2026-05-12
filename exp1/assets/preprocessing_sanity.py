"""Integration sanity checks for mesh preprocessing outputs.

The helpers in this module intentionally validate files on disk.  They are not
rendering checks; they exercise asset discovery, normalization, texture-condition
GLB export, and report generation before the Blender/model stages run.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from exp1.assets.discover import discover_assets_from_directory
from exp1.assets.normalize import normalize_asset_manifest
from exp1.assets.shapenet import discover_huggingface_shapenet_assets
from exp1.metadata.schema import VALID_TEXTURE_CONDITIONS
from src.utils.io import write_jsonl


DATASET_OUTPUT_NAMES = {
    "objaverse": "objaverse",
    "shapenet": "shapenetcore",
    "shapenetcore": "shapenetcore",
}
TEXTURELESS_SOURCE_DATASETS = {"synthetic_primitives"}
TEXTURELESS_RAW_EXTENSIONS = {".off"}

SHAPENETCORE_EXPECTED_LAYOUTS = (
    "local Hugging Face ShapeNetCore snapshot, e.g. root/<synset_id>.zip",
    "ShapeNetCore-style tree, e.g. root/<synset_id>/<model_id>/models/*.obj",
    "optional category ZIP snapshots when --shapenetcore-extract-archives is set",
)
OBJAVERSE_EXPECTED_LAYOUTS = (
    "local Objaverse mesh tree with .glb/.gltf/.obj/.fbx/.ply assets",
    "optional split/category folders, e.g. root/train/chair/<uid>.glb",
)


def safe_component(value: Any) -> str:
    """Return a filesystem-safe path component."""
    text = str(value or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "unknown"


def report_dataset_name(source_dataset: Any) -> str:
    """Map internal dataset names to report/output names requested by Exp. 1."""
    key = str(source_dataset or "").strip().lower()
    return DATASET_OUTPUT_NAMES.get(key, safe_component(key or "unknown"))


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


def source_has_photorealistic_material(
    row: Mapping[str, Any],
    source_info: Mapping[str, Any],
) -> bool:
    """Conservatively decide whether a source has original material data."""
    source_dataset = str(row.get("source_dataset", "")).strip().lower()
    raw_path = Path(str(row.get("raw_mesh_path", "")))
    if source_dataset in TEXTURELESS_SOURCE_DATASETS:
        return False
    if raw_path.suffix.lower() in TEXTURELESS_RAW_EXTENSIONS:
        return False

    for key in (
        "has_photorealistic_material",
        "photorealistic_material_available",
        "has_imported_material",
    ):
        if key in row:
            value = _truthy(row.get(key))
            if value is not None:
                return value

    if bool(source_info.get("has_texture")):
        return True
    if raw_path.suffix.lower() in {".glb", ".gltf", ".fbx", ".obj"}:
        return bool(source_info.get("has_material"))
    return False


def parse_texture_conditions(values: Optional[Sequence[str]]) -> List[str]:
    """Parse comma and/or whitespace separated texture-condition values."""
    if not values:
        return list(VALID_TEXTURE_CONDITIONS)
    parsed: List[str] = []
    for value in values:
        for part in str(value).split(","):
            condition = part.strip()
            if condition:
                parsed.append(condition)
    invalid = [c for c in parsed if c not in set(VALID_TEXTURE_CONDITIONS)]
    if invalid:
        raise ValueError(
            "Invalid texture conditions: "
            + ", ".join(invalid)
            + ". Expected one or more of: "
            + ", ".join(VALID_TEXTURE_CONDITIONS)
        )
    return parsed


def select_balanced_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Select a deterministic category-balanced subset of asset rows."""
    if limit <= 0:
        return []

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        category = str(row.get("category", "unknown"))
        groups.setdefault(category, []).append(dict(row))

    rng = random.Random(int(seed))
    categories = sorted(groups)
    rng.shuffle(categories)
    for category in categories:
        groups[category].sort(
            key=lambda r: (
                str(r.get("split", "")),
                str(r.get("object_id", "")),
                str(r.get("raw_mesh_path", "")),
            )
        )
        rng.shuffle(groups[category])

    selected: List[Dict[str, Any]] = []
    while categories and len(selected) < limit:
        next_categories: List[str] = []
        for category in categories:
            if groups[category] and len(selected) < limit:
                selected.append(groups[category].pop(0))
            if groups[category]:
                next_categories.append(category)
        categories = next_categories
    return selected


def discover_objaverse_for_sanity(
    root: Path,
    *,
    limit: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Discover and sample local Objaverse assets."""
    warnings: List[str] = []
    if not root.is_dir():
        return [], [f"Objaverse root does not exist: {root}"]
    rows = discover_assets_from_directory(root, source_dataset="objaverse")
    if not rows:
        return [], [
            "No Objaverse meshes discovered under "
            f"{root}. Expected one of: {', '.join(OBJAVERSE_EXPECTED_LAYOUTS)}"
        ]
    sampled = select_balanced_subset(rows, limit=limit, seed=seed)
    categories = {str(row.get("category", "unknown")) for row in sampled}
    if len(categories) < 2 and len({str(row.get("category")) for row in rows}) >= 2:
        warnings.append(
            "Objaverse sample contains fewer than two categories despite multiple "
            "categories being available."
        )
    return sampled, warnings


def discover_shapenetcore_for_sanity(
    root: Path,
    *,
    limit: int,
    seed: int,
    categories: Optional[Sequence[str]] = None,
    extract_archives: bool = False,
    extracted_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Discover and sample local ShapeNetCore/HF ShapeNet assets."""
    warnings: List[str] = []
    if not root.is_dir():
        return [], [f"ShapeNetCore root does not exist: {root}"]
    try:
        rows = discover_huggingface_shapenet_assets(
            local_dir=root,
            download=False,
            categories=categories,
            extract_archives=extract_archives,
            extracted_dir=extracted_dir,
            source_dataset="shapenet",
        )
    except Exception as exc:
        return [], [f"ShapeNetCore discovery failed: {exc}"]

    if not rows:
        zip_hint = ""
        if any(root.rglob("*.zip")) and not extract_archives:
            zip_hint = (
                " ZIP archives were found; rerun with "
                "--shapenetcore-extract-archives and optionally "
                "--shapenetcore-category to limit extraction."
            )
        return [], [
            "No ShapeNetCore meshes discovered under "
            f"{root}. Expected one of: {', '.join(SHAPENETCORE_EXPECTED_LAYOUTS)}."
            + zip_hint
        ]
    sampled = select_balanced_subset(rows, limit=limit, seed=seed)
    categories_found = {str(row.get("category", "unknown")) for row in sampled}
    if (
        len(categories_found) < 2
        and len({str(row.get("category")) for row in rows}) >= 2
    ):
        warnings.append(
            "ShapeNetCore sample contains fewer than two categories despite "
            "multiple categories being available."
        )
    return sampled, warnings


def _load_any_mesh(path: Path) -> Any:
    import trimesh

    return trimesh.load(path, force=None, process=False)


def _iter_geometries(loaded: Any) -> Iterable[Any]:
    import trimesh

    if isinstance(loaded, trimesh.Scene):
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                yield geom
    elif isinstance(loaded, trimesh.Trimesh):
        yield loaded


def _as_single_mesh(path: Path) -> Any:
    import trimesh

    loaded = _load_any_mesh(path)
    meshes = [mesh.copy() for mesh in _iter_geometries(loaded)]
    if not meshes:
        raise ValueError(f"No mesh geometry found in {path}")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _mesh_bounds(meshes: Sequence[Any]) -> Dict[str, Any]:
    bounds = []
    for mesh in meshes:
        mesh_bounds = getattr(mesh, "bounds", None)
        if mesh_bounds is not None:
            arr = np.asarray(mesh_bounds, dtype=float)
            if arr.shape == (2, 3) and np.isfinite(arr).all():
                bounds.append(arr)
    if not bounds:
        return {}
    stacked = np.stack(bounds, axis=0)
    low = stacked[:, 0, :].min(axis=0)
    high = stacked[:, 1, :].max(axis=0)
    extents = high - low
    return {
        "bounds_min": [float(v) for v in low],
        "bounds_max": [float(v) for v in high],
        "extent": [float(v) for v in extents],
        "max_extent": float(extents.max()),
    }


def _texture_image(material: Any) -> Any:
    for attr in ("baseColorTexture", "image"):
        image = getattr(material, attr, None)
        if image is not None:
            return image
    return None


def _image_sha1(image: Any) -> str:
    if image is None:
        return ""
    try:
        arr = np.asarray(image.convert("RGBA"))
    except AttributeError:
        arr = np.asarray(image)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _material_base_color(material: Any) -> Optional[List[int]]:
    color = getattr(material, "baseColorFactor", None)
    if color is None:
        color = getattr(material, "diffuse", None)
    if color is None:
        color = getattr(material, "main_color", None)
    if color is None:
        return None
    arr = np.asarray(color).reshape(-1)
    if arr.size < 3:
        return None
    return [int(v) for v in arr[:4]]


def inspect_mesh_file(path: Path) -> Dict[str, Any]:
    """Reload a mesh file and summarize physical geometry/material fields."""
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "file_size_bytes": int(path.stat().st_size) if path.is_file() else 0,
        "reload_ok": False,
        "geometry_count": 0,
        "vertex_count": 0,
        "face_count": 0,
        "has_geometry": False,
        "material_count": 0,
        "texture_count": 0,
        "has_material": False,
        "has_texture": False,
        "visual_kinds": [],
        "material_base_colors": [],
        "texture_image_sizes": [],
        "texture_image_sha1": [],
        "bounds": {},
        "error": "",
    }
    if not info["exists"]:
        info["error"] = "file_missing"
        return info
    if info["file_size_bytes"] <= 0:
        info["error"] = "file_empty"
        return info

    try:
        loaded = _load_any_mesh(path)
        meshes = list(_iter_geometries(loaded))
        info["geometry_count"] = len(meshes)
        vertex_count = 0
        face_count = 0
        for mesh in meshes:
            vertices = getattr(mesh, "vertices", None)
            faces = getattr(mesh, "faces", None)
            vertex_count += len(vertices) if vertices is not None else 0
            face_count += len(faces) if faces is not None else 0
        info["vertex_count"] = int(vertex_count)
        info["face_count"] = int(face_count)
        info["has_geometry"] = bool(
            info["geometry_count"] > 0
            and info["vertex_count"] > 0
            and info["face_count"] > 0
        )
        info["bounds"] = _mesh_bounds(meshes)

        visual_kinds: List[str] = []
        colors: List[List[int]] = []
        image_sizes: List[List[int]] = []
        image_hashes: List[str] = []
        material_count = 0
        texture_count = 0
        for mesh in meshes:
            visual = getattr(mesh, "visual", None)
            if visual is None:
                continue
            kind = str(getattr(visual, "kind", type(visual).__name__))
            visual_kinds.append(kind)
            material = getattr(visual, "material", None)
            if material is not None:
                material_count += 1
                color = _material_base_color(material)
                if color is not None:
                    colors.append(color)
                image = _texture_image(material)
                if image is not None:
                    texture_count += 1
                    image_sizes.append([int(image.size[0]), int(image.size[1])])
                    image_hashes.append(_image_sha1(image))
            elif kind == "face" or kind == "vertex":
                material_count += 1

        info["reload_ok"] = True
        info["material_count"] = int(material_count)
        info["texture_count"] = int(texture_count)
        info["has_material"] = bool(material_count > 0)
        info["has_texture"] = bool(texture_count > 0)
        info["visual_kinds"] = sorted(set(visual_kinds))
        info["material_base_colors"] = colors
        info["texture_image_sizes"] = image_sizes
        info["texture_image_sha1"] = image_hashes
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _assign_flat_material(mesh: Any, *, color_rgba: Sequence[int]) -> Any:
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    out = mesh.copy()
    material = PBRMaterial(
        name="exp1_flat_textureless",
        baseColorFactor=list(color_rgba),
        roughnessFactor=0.55,
        metallicFactor=0.0,
    )
    out.visual = TextureVisuals(material=material)
    return out


def _uv_from_vertices(mesh: Any) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        return np.zeros((0, 2), dtype=float)
    axes = np.argsort(np.ptp(vertices, axis=0))[-2:]
    coords = vertices[:, axes]
    min_xy = coords.min(axis=0)
    extent_xy = np.ptp(coords, axis=0)
    extent_xy = np.where(extent_xy <= 1e-8, 1.0, extent_xy)
    return (coords - min_xy) / extent_xy


def _assign_noise_texture(
    mesh: Any,
    *,
    seed: int,
    size: int = 64,
) -> Tuple[Any, Dict[str, Any]]:
    from PIL import Image
    from trimesh.visual.material import SimpleMaterial
    from trimesh.visual.texture import TextureVisuals

    out = mesh.copy()
    rng = np.random.default_rng(int(seed))
    noise = rng.integers(0, 256, size=(int(size), int(size), 3), dtype=np.uint8)
    image = Image.fromarray(noise)
    material = SimpleMaterial(image=image, diffuse=[255, 255, 255, 255])
    out.visual = TextureVisuals(uv=_uv_from_vertices(out), material=material)
    return out, {
        "random_noise_texture_type": "embedded_image",
        "random_noise_image_size": [int(size), int(size)],
        "random_noise_image_sha1": hashlib.sha1(noise.tobytes()).hexdigest(),
    }


def export_texture_variant(
    row: Mapping[str, Any],
    *,
    texture_condition: str,
    output_root: Path,
    seed: int,
) -> Dict[str, Any]:
    """Export one physical GLB variant for a normalized asset row."""
    if texture_condition not in set(VALID_TEXTURE_CONDITIONS):
        raise ValueError(f"Unknown texture condition: {texture_condition}")

    source_dataset = report_dataset_name(row.get("source_dataset"))
    object_id = safe_component(row.get("object_id"))
    category = safe_component(row.get("category"))
    output_path = (
        output_root / source_dataset / category / object_id / f"{texture_condition}.glb"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_path = Path(str(row.get("normalized_mesh_path", "")))
    if not normalized_path.is_file():
        raise FileNotFoundError(f"Missing normalized GLB: {normalized_path}")

    material_meta: Dict[str, Any] = {
        "material_condition": texture_condition,
        "texture_seed_used": int(seed),
        "random_texture_path": "",
    }

    source_info = inspect_mesh_file(Path(str(row.get("raw_mesh_path", ""))))
    normalized_info = inspect_mesh_file(normalized_path)
    source_material_available = source_has_photorealistic_material(
        row,
        source_info,
    )
    source_texture_available = bool(source_info.get("has_texture"))

    if texture_condition == "photorealistic":
        if source_material_available and normalized_info.get("has_material"):
            loaded = _load_any_mesh(normalized_path)
            loaded.export(output_path)
            material_meta.update(
                {
                    "material_mode": "photorealistic",
                    "material_status": "preserved_original",
                    "photorealistic_material_status": "preserved",
                    "photorealistic_fallback_reason": "",
                    "source_material_available": True,
                    "source_texture_available": source_texture_available,
                }
            )
        else:
            mesh = _as_single_mesh(normalized_path)
            mesh = _assign_flat_material(mesh, color_rgba=[173, 173, 173, 255])
            mesh.export(output_path)
            reason = (
                "source_has_no_detectable_material"
                if not source_material_available
                else "normalized_glb_has_no_material"
            )
            material_meta.update(
                {
                    "material_mode": "photorealistic_fallback",
                    "material_status": "fallback_missing_original",
                    "photorealistic_material_status": "fallback_missing_original",
                    "photorealistic_fallback_reason": reason,
                    "photorealistic_fallback_color_rgb": [173, 173, 173],
                    "source_material_available": source_material_available,
                    "source_texture_available": source_texture_available,
                }
            )
        return {"output_path": str(output_path), **material_meta}

    mesh = _as_single_mesh(normalized_path)
    if texture_condition == "flat":
        mesh = _assign_flat_material(mesh, color_rgba=[158, 158, 158, 255])
        mesh.export(output_path)
        material_meta.update(
            {
                "material_mode": "flat",
                "material_status": "flat_override",
                "flat_color_rgb": [158, 158, 158],
                "source_material_available": source_material_available,
                "source_texture_available": source_texture_available,
            }
        )
        return {"output_path": str(output_path), **material_meta}

    mesh, noise_meta = _assign_noise_texture(mesh, seed=seed)
    mesh.export(output_path)
    material_meta.update(
        {
            "material_mode": "random_noise",
            "material_status": "random_noise_override",
            "source_material_available": source_material_available,
            "source_texture_available": source_texture_available,
            **noise_meta,
        }
    )
    return {"output_path": str(output_path), **material_meta}


def validate_texture_variant(
    *,
    texture_condition: str,
    material_meta: Mapping[str, Any],
    output_path: Path,
    strict: bool,
) -> Dict[str, Any]:
    """Validate a physical texture-condition GLB on disk."""
    inspection = inspect_mesh_file(output_path)
    checks: List[str] = []
    warnings: List[str] = []
    failures: List[str] = []

    def require(name: str, condition: bool, message: str) -> None:
        checks.append(name)
        if not condition:
            failures.append(message)

    require("file_exists", bool(inspection["exists"]), "GLB file was not written")
    require(
        "file_nonzero",
        int(inspection["file_size_bytes"]) > 0,
        "GLB file is empty",
    )
    require("reload_ok", bool(inspection["reload_ok"]), "GLB failed to reload")
    require("has_geometry", bool(inspection["has_geometry"]), "GLB has no geometry")

    if texture_condition == "photorealistic":
        source_available = bool(material_meta.get("source_material_available"))
        if source_available:
            require(
                "preserves_source_material",
                bool(inspection["has_material"]),
                "Photorealistic source material was expected but not present",
            )
            if bool(material_meta.get("source_texture_available")):
                require(
                    "preserves_source_texture",
                    bool(inspection["has_texture"]),
                    "Photorealistic source texture was expected but not present",
                )
        else:
            warnings.append(
                "Photorealistic/original material unavailable for this source; "
                "exported documented neutral fallback instead."
            )
            if strict:
                checks.append("photorealistic_missing_source_material_warned")
    elif texture_condition == "flat":
        require(
            "flat_has_material",
            bool(inspection["has_material"]),
            "Flat material missing",
        )
        require(
            "flat_has_no_texture",
            not bool(inspection["has_texture"]),
            "Flat textureless GLB retained an image texture",
        )
        colors = inspection.get("material_base_colors") or []
        require(
            "flat_constant_color",
            bool(colors)
            and all(list(color[:3]) == [158, 158, 158] for color in colors),
            "Flat material base color is not the configured constant color",
        )
    elif texture_condition == "random_noise":
        require(
            "random_noise_has_texture",
            bool(inspection["has_texture"]),
            "Random-noise GLB has no embedded texture",
        )
        texture_hashes = inspection.get("texture_image_sha1") or []
        require(
            "random_noise_not_flat",
            bool(texture_hashes),
            "Random-noise material is indistinguishable from flat material",
        )

    return {
        "inspection": inspection,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "passed": not failures,
    }


def build_artifact_record(
    row: Mapping[str, Any],
    *,
    texture_condition: str,
    variant_meta: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Combine source, normalization, material, and validation metadata."""
    normalized_bounds = {
        "extent": [
            row.get("normalized_extent_x"),
            row.get("normalized_extent_y"),
            row.get("normalized_extent_z"),
        ],
        "max_extent": row.get("normalized_max_extent"),
    }
    return {
        "source_dataset": report_dataset_name(row.get("source_dataset")),
        "pipeline_source_dataset": row.get("source_dataset"),
        "source_object_id": row.get("object_id"),
        "shapenet_model_id": row.get("shapenet_model_id", ""),
        "shapenet_synset_id": row.get("shapenet_synset_id", ""),
        "category_label": row.get("category"),
        "split": row.get("split"),
        "original_mesh_path": row.get("raw_mesh_path"),
        "normalized_mesh_path": row.get("normalized_mesh_path"),
        "output_glb_path": variant_meta.get("output_path"),
        "texture_condition": texture_condition,
        "material_texture_metadata": {
            key: value for key, value in variant_meta.items() if key != "output_path"
        },
        "normalization": {
            "center": [
                row.get("normalization_center_x"),
                row.get("normalization_center_y"),
                row.get("normalization_center_z"),
            ],
            "scale": row.get("normalization_scale"),
            "bounds": normalized_bounds,
            "canonical_orientation": row.get(
                "canonical_orientation",
                "not_produced_by_current_preprocessing",
            ),
        },
        "geometry_labels": {
            "vertex_count": row.get("vertex_count"),
            "face_count": row.get("face_count"),
            "normalized_extent_x": row.get("normalized_extent_x"),
            "normalized_extent_y": row.get("normalized_extent_y"),
            "normalized_extent_z": row.get("normalized_extent_z"),
            "normalized_max_extent": row.get("normalized_max_extent"),
        },
        "camera_render_labels": row.get(
            "camera_render_labels",
            "not_generated_by_preprocessing_sanity_check",
        ),
        "validation": validation,
        "failed_or_skipped_checks": list(validation.get("failures", []))
        + list(validation.get("warnings", [])),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Preprocessing Sanity Check Report",
        "",
        "## What data/labels were produced, and where can I find them?",
        "",
        f"- GLB variants: `{summary['glb_output_dir']}`",
        f"- Metadata: `{summary['metadata_output_dir']}`",
        f"- JSON report: `{summary['json_report_path']}`",
        f"- Markdown report: `{summary['markdown_report_path']}`",
        f"- Raw asset manifest: `{summary['raw_asset_manifest_path']}`",
        f"- Normalized asset manifest: `{summary['normalized_asset_manifest_path']}`",
        "",
        "Each artifact row records dataset, object/model id, category, split, "
        "source mesh path, normalized GLB path, texture-condition GLB path, "
        "material/texture metadata, normalization scale/bounds, available "
        "geometry labels, and validation failures or warnings.",
        "",
        "## Summary",
        "",
        "- ShapeNetCore objects processed: "
        f"{summary['num_shapenetcore_objects_processed']}",
        f"- Objaverse objects processed: {summary['num_objaverse_objects_processed']}",
        f"- GLB files written: {summary['num_glb_files_written']}",
        "- Texture-condition variants attempted: "
        f"{summary['num_texture_condition_variants_attempted']}",
        "- Texture-condition variants written: "
        f"{summary['num_texture_condition_variants_written']}",
        f"- Failures: {summary['num_failures']}",
        f"- Warnings: {summary['num_warnings']}",
        "",
        "## Expected Input Layouts",
        "",
        "- ShapeNetCore: " + "; ".join(SHAPENETCORE_EXPECTED_LAYOUTS),
        "- Objaverse: " + "; ".join(OBJAVERSE_EXPECTED_LAYOUTS),
        "",
        "## Output Directories",
        "",
    ]
    for directory in summary["output_directories"]:
        lines.append(f"- `{directory}`")

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in report["failures"]:
            lines.append(f"- {failure}")

    lines.extend(["", "## Artifacts", ""])
    for artifact in report["artifacts"]:
        status = "PASS" if artifact["validation"]["passed"] else "FAIL"
        lines.extend(
            [
                f"### {artifact['source_dataset']} / "
                f"{artifact['category_label']} / "
                f"{artifact['source_object_id']} / "
                f"{artifact['texture_condition']} [{status}]",
                "",
                f"- Source mesh: `{artifact['original_mesh_path']}`",
                f"- Normalized mesh: `{artifact['normalized_mesh_path']}`",
                f"- Output GLB: `{artifact['output_glb_path']}`",
                f"- Split: `{artifact['split']}`",
                f"- Material status: "
                f"`{artifact['material_texture_metadata'].get('material_status')}`",
                f"- Normalization scale: `{artifact['normalization']['scale']}`",
                f"- Geometry labels: vertex_count="
                f"`{artifact['geometry_labels'].get('vertex_count')}`, "
                f"face_count=`{artifact['geometry_labels'].get('face_count')}`",
                f"- Failed/skipped checks: "
                f"{artifact['failed_or_skipped_checks'] or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: Mapping[str, Any], *, metadata_dir: Path) -> Tuple[Path, Path]:
    """Write JSON and Markdown reports and return their paths."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    json_path = metadata_dir / "preprocessing_report.json"
    markdown_path = metadata_dir / "preprocessing_report.md"

    report_dict = dict(report)
    report_dict["summary"] = dict(report_dict["summary"])
    report_dict["summary"]["json_report_path"] = str(json_path)
    report_dict["summary"]["markdown_report_path"] = str(markdown_path)

    json_path.write_text(json.dumps(report_dict, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_report_markdown(report_dict), encoding="utf-8")
    return json_path, markdown_path


def run_preprocessing_sanity_check(
    *,
    shapenetcore_root: Optional[Path],
    objaverse_root: Optional[Path],
    output_dir: Path,
    num_objects_per_dataset: int,
    texture_conditions: Sequence[str],
    seed: int,
    strict: bool,
    shapenetcore_categories: Optional[Sequence[str]] = None,
    shapenetcore_extract_archives: bool = False,
    target_extent: float = 1.0,
) -> Dict[str, Any]:
    """Run the preprocessing sanity check and return the report dictionary."""
    output_dir = output_dir.expanduser().resolve()
    glb_dir = output_dir / "glbs"
    metadata_dir = output_dir / "metadata"
    normalized_root = glb_dir / "normalized"
    variant_root = glb_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    glb_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    failures: List[str] = []
    discovered: List[Dict[str, Any]] = []
    selected_by_dataset: Dict[str, List[Dict[str, Any]]] = {
        "shapenetcore": [],
        "objaverse": [],
    }

    if shapenetcore_root is None:
        msg = "ShapeNetCore root not provided; pass --shapenetcore-root."
        warnings.append(msg)
        if strict:
            failures.append(msg)
    else:
        rows, dataset_warnings = discover_shapenetcore_for_sanity(
            shapenetcore_root.expanduser(),
            limit=num_objects_per_dataset,
            seed=seed + 17,
            categories=shapenetcore_categories,
            extract_archives=shapenetcore_extract_archives,
            extracted_dir=metadata_dir / "shapenetcore_extracted",
        )
        warnings.extend(dataset_warnings)
        selected_by_dataset["shapenetcore"] = rows
        discovered.extend(rows)
        if not rows:
            failures.append("No ShapeNetCore objects were selected for preprocessing.")

    if objaverse_root is None:
        msg = "Objaverse root not provided; pass --objaverse-root."
        warnings.append(msg)
        if strict:
            failures.append(msg)
    else:
        rows, dataset_warnings = discover_objaverse_for_sanity(
            objaverse_root.expanduser(),
            limit=num_objects_per_dataset,
            seed=seed + 29,
        )
        warnings.extend(dataset_warnings)
        selected_by_dataset["objaverse"] = rows
        discovered.extend(rows)
        if not rows:
            failures.append("No Objaverse objects were selected for preprocessing.")

    if not discovered:
        failures.append(
            "No assets were discovered from any dataset; cannot validate preprocessing."
        )

    raw_manifest_path = metadata_dir / "selected_raw_assets.jsonl"
    normalized_manifest_path = metadata_dir / "selected_normalized_assets.jsonl"
    write_jsonl(raw_manifest_path, discovered)

    normalized_rows = normalize_asset_manifest(
        discovered,
        output_root=normalized_root,
        target_extent=target_extent,
        overwrite=True,
        fail_fast=False,
    )
    write_jsonl(normalized_manifest_path, normalized_rows)

    artifacts: List[Dict[str, Any]] = []
    for row in normalized_rows:
        if row.get("asset_status") != "normalized":
            failures.append(
                f"Normalization failed for {row.get('object_id')}: "
                f"{row.get('asset_error_message')}"
            )
            continue
        for condition in texture_conditions:
            texture_seed = int(
                hashlib.sha1(
                    f"{seed}|{row.get('object_id')}|{condition}".encode("utf-8")
                ).hexdigest()[:8],
                16,
            )
            try:
                variant_meta = export_texture_variant(
                    row,
                    texture_condition=condition,
                    output_root=variant_root,
                    seed=texture_seed,
                )
                validation = validate_texture_variant(
                    texture_condition=condition,
                    material_meta=variant_meta,
                    output_path=Path(str(variant_meta["output_path"])),
                    strict=strict,
                )
            except Exception as exc:
                variant_meta = {
                    "output_path": "",
                    "material_condition": condition,
                    "material_status": "export_failed",
                    "texture_seed_used": texture_seed,
                }
                validation = {
                    "inspection": {},
                    "checks": [],
                    "warnings": [],
                    "failures": [str(exc)],
                    "passed": False,
                }

            artifacts.append(
                build_artifact_record(
                    row,
                    texture_condition=condition,
                    variant_meta=variant_meta,
                    validation=validation,
                )
            )
            warnings.extend(str(w) for w in validation.get("warnings", []))
            failures.extend(str(f) for f in validation.get("failures", []))

    output_directories = sorted(
        {
            str(path.parent)
            for artifact in artifacts
            for path in [Path(str(artifact.get("output_glb_path", "")))]
            if str(path) and str(path) != "."
        }
        | {str(glb_dir), str(metadata_dir)}
    )
    glb_written = [
        artifact
        for artifact in artifacts
        if artifact.get("output_glb_path")
        and Path(str(artifact["output_glb_path"])).is_file()
    ]
    report: Dict[str, Any] = {
        "summary": {
            "strict": bool(strict),
            "seed": int(seed),
            "texture_conditions": list(texture_conditions),
            "num_shapenetcore_objects_processed": len(
                selected_by_dataset["shapenetcore"]
            ),
            "num_objaverse_objects_processed": len(
                selected_by_dataset["objaverse"]
            ),
            "num_glb_files_written": len(glb_written),
            "num_texture_condition_variants_attempted": len(artifacts),
            "num_texture_condition_variants_written": len(glb_written),
            "num_failures": len(failures),
            "num_warnings": len(warnings),
            "output_dir": str(output_dir),
            "glb_output_dir": str(glb_dir),
            "metadata_output_dir": str(metadata_dir),
            "raw_asset_manifest_path": str(raw_manifest_path),
            "normalized_asset_manifest_path": str(normalized_manifest_path),
            "json_report_path": str(metadata_dir / "preprocessing_report.json"),
            "markdown_report_path": str(metadata_dir / "preprocessing_report.md"),
            "output_directories": output_directories,
        },
        "expected_input_layouts": {
            "shapenetcore": list(SHAPENETCORE_EXPECTED_LAYOUTS),
            "objaverse": list(OBJAVERSE_EXPECTED_LAYOUTS),
        },
        "warnings": sorted(set(warnings)),
        "failures": failures,
        "artifacts": artifacts,
    }
    json_path, markdown_path = write_report(report, metadata_dir=metadata_dir)
    report["summary"]["json_report_path"] = str(json_path)
    report["summary"]["markdown_report_path"] = str(markdown_path)
    return report

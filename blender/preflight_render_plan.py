"""
Blender-side preflight checks for an Experiment 1 render plan.

This script intentionally uses only Blender's Python standard library plus
``bpy``. Do not import project ML/data dependencies such as pandas here.

Example:
  blender --background --python blender/preflight_render_plan.py -- \
      data/exp1_sanity/render_plan.jsonl --max-imports 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import bpy  # type: ignore[reportMissingImports]


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mesh_utils import import_mesh  # noqa: E402


REQUIRED_RENDER_COLUMNS = (
    "render_id",
    "object_id",
    "source_dataset",
    "category",
    "split",
    "raw_mesh_path",
    "normalized_mesh_path",
    "texture_condition",
    "texture_seed",
    "camera_distance",
    "camera_azimuth_deg",
    "camera_elevation_deg",
    "camera_fov_deg",
    "object_scale",
    "light_type",
    "light_azimuth_deg",
    "light_elevation_deg",
    "light_intensity",
    "rgb_path",
    "depth_path",
    "normal_path",
    "mask_path",
    "render_seed",
    "render_status",
    "qc_error_message",
)
VALID_SPLITS = {"train", "val", "test"}
VALID_TEXTURE_CONDITIONS = {"photorealistic", "flat", "random_noise"}
CONTROL_FIELDS = (
    "object_id",
    "source_dataset",
    "split",
    "normalized_mesh_path",
    "camera_distance",
    "camera_azimuth_deg",
    "camera_elevation_deg",
    "camera_fov_deg",
    "object_scale",
    "light_type",
    "light_azimuth_deg",
    "light_elevation_deg",
    "light_intensity",
    "render_seed",
    "grid_index",
)


def _argv_after_blender_separator() -> List[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "render_plan",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data/exp1_sanity/render_plan.jsonl",
        help="Path to a JSONL, JSON, or CSV render plan.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root used to resolve relative plan and mesh paths.",
    )
    parser.add_argument(
        "--max-imports",
        type=int,
        default=3,
        help="Number of rows whose meshes should be imported in Blender.",
    )
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="Only validate manifest fields and mesh paths; do not import meshes.",
    )
    return parser.parse_args(list(argv))


def _resolve_path(path: Path, project_root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "rows" in payload:
        payload = payload["rows"]
    if not isinstance(payload, list):
        raise ValueError("JSON render plan must be a list or an object with 'rows'")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError("JSON render plan rows must all be objects")
    return list(payload)


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_render_plan(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise ValueError(
        f"Unsupported render plan extension '{path.suffix}'. "
        "Use .jsonl, .json, or .csv for Blender preflight."
    )


def _missing_columns(row: Dict[str, Any], columns: Iterable[str]) -> List[str]:
    missing = []
    for column in columns:
        if column not in row or row[column] is None:
            missing.append(column)
    return missing


def validate_required_columns(rows: Sequence[Dict[str, Any]]) -> None:
    for idx, row in enumerate(rows):
        missing = _missing_columns(row, REQUIRED_RENDER_COLUMNS)
        if missing:
            raise ValueError(
                f"Row {idx} is missing required columns: {', '.join(missing)}"
            )


def validate_duplicate_render_ids(rows: Sequence[Dict[str, Any]]) -> None:
    counts = Counter(str(row["render_id"]) for row in rows)
    duplicates = sorted(render_id for render_id, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:8])
        if len(duplicates) > 8:
            preview += f", ... +{len(duplicates) - 8} more"
        raise ValueError(f"Duplicate render_id values: {preview}")


def validate_enum_values(rows: Sequence[Dict[str, Any]]) -> None:
    invalid_splits = sorted(
        {str(row["split"]) for row in rows if str(row["split"]) not in VALID_SPLITS}
    )
    if invalid_splits:
        raise ValueError(f"Invalid split values: {', '.join(invalid_splits)}")

    invalid_textures = sorted(
        {
            str(row["texture_condition"])
            for row in rows
            if str(row["texture_condition"]) not in VALID_TEXTURE_CONDITIONS
        }
    )
    if invalid_textures:
        raise ValueError(
            f"Invalid texture_condition values: {', '.join(invalid_textures)}"
        )


def validate_texture_control_groups(rows: Sequence[Dict[str, Any]]) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("texture_control_group_id")
        if group_id is not None and str(group_id) != "":
            groups[str(group_id)].append(row)

    if not groups:
        print("No texture_control_group_id column found; skipping control-group check.")
        return

    for group_id, group_rows in groups.items():
        textures = {str(row["texture_condition"]) for row in group_rows}
        if textures != VALID_TEXTURE_CONDITIONS:
            raise ValueError(
                f"Texture control group {group_id} has textures "
                f"{sorted(textures)}, expected {sorted(VALID_TEXTURE_CONDITIONS)}"
            )

        reference = group_rows[0]
        for row in group_rows[1:]:
            mismatched = [
                field
                for field in CONTROL_FIELDS
                if field in reference
                and field in row
                and str(reference[field]) != str(row[field])
            ]
            if mismatched:
                raise ValueError(
                    f"Texture control group {group_id} has mismatched fields: "
                    + ", ".join(mismatched)
                )


def validate_mesh_paths(rows: Sequence[Dict[str, Any]], project_root: Path) -> None:
    missing = []
    for row in rows:
        mesh_path = _resolve_path(Path(str(row["normalized_mesh_path"])), project_root)
        if not mesh_path.is_file():
            missing.append((row["render_id"], str(mesh_path)))

    if missing:
        preview = "; ".join(
            f"{render_id}: {mesh_path}" for render_id, mesh_path in missing[:8]
        )
        if len(missing) > 8:
            preview += f"; ... +{len(missing) - 8} more"
        raise FileNotFoundError(f"Missing normalized meshes: {preview}")


def import_mesh_samples(
    rows: Sequence[Dict[str, Any]],
    *,
    project_root: Path,
    max_imports: int,
) -> None:
    for row in rows[: max(0, max_imports)]:
        render_id = str(row["render_id"])
        mesh_path = _resolve_path(Path(str(row["normalized_mesh_path"])), project_root)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        import_mesh(str(mesh_path))
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        if not mesh_objects:
            raise RuntimeError(f"No mesh objects after importing {mesh_path}")
        print(f"OK render row {render_id} imports mesh {mesh_path.name}")


def main() -> None:
    args = parse_args(_argv_after_blender_separator())
    project_root = args.project_root.expanduser().resolve()
    render_plan = _resolve_path(args.render_plan, project_root)
    if not render_plan.is_file():
        raise FileNotFoundError(f"Missing render plan: {render_plan}")

    rows = load_render_plan(render_plan)
    if not rows:
        raise ValueError(f"Render plan has no rows: {render_plan}")

    validate_required_columns(rows)
    validate_duplicate_render_ids(rows)
    validate_enum_values(rows)
    validate_texture_control_groups(rows)
    validate_mesh_paths(rows, project_root)

    if not args.skip_imports:
        import_mesh_samples(rows, project_root=project_root, max_imports=args.max_imports)

    texture_counts = Counter(str(row["texture_condition"]) for row in rows)
    print(f"Checked {len(rows)} render rows from {render_plan}")
    print(f"Texture counts: {dict(sorted(texture_counts.items()))}")
    print("Render plan Blender preflight passed.")


if __name__ == "__main__":
    main()

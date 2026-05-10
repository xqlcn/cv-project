#!/usr/bin/env python3
"""Render Experiment 1 rows in Blender.

Run with Blender:

  blender --background --python scripts/render_blender.py -- \\
      --config configs/exp1_smoke.yaml \\
      --chunk data/exp1/manifests/render_chunks/chunk_000.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

try:
    import bpy  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_paths(project_root: Path) -> None:
    for path in (project_root, project_root / "blender"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


_ensure_paths(PROJECT_ROOT)

from exp1.rendering.camera import camera_metadata, setup_camera  # noqa: E402
from exp1.rendering.lighting import setup_lighting  # noqa: E402
from exp1.rendering.materials import apply_materials  # noqa: E402
from exp1.rendering.passes import buffer_shapes, render_geometry_buffers  # noqa: E402


DEFAULT_CONFIG: Dict[str, Any] = {
    "render": {
        "engine": "CYCLES",
        "resolution": [224, 224],
        "samples": 64,
        "use_gpu": True,
        "transparent_background": False,
        "file_format": "PNG",
        "fail_fast": False,
        "camera": {"fov_deg": 50.0, "clip_start": 0.01, "clip_end": 1000.0},
        "world": {"bg_color": [0.05, 0.05, 0.06, 1.0]},
        "output_contract": {
            "rgb_filename": "rgb.png",
            "depth_filename": "depth.npy",
            "normal_filename": "normal_camera.npy",
            "mask_filename": "mask.npy",
            "metadata_filename": "render_meta.json",
        },
    },
    "textures": {
        "flat": {"color_rgb": [0.62, 0.62, 0.62]},
        "photorealistic": {
            "preserve_imported_materials": True,
            "fallback_color_rgb": [0.68, 0.68, 0.68],
            "textureless_source_datasets": ["modelnet40", "synthetic_primitives"],
        },
        "random_noise": {"scale_range": [8.0, 20.0], "detail": 6.0},
    },
    "lighting_grid": {"fill_intensity": 0.35},
}


def _recursive_update(
    base: Dict[str, Any],
    update: Mapping[str, Any],
) -> Dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _recursive_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_simple_yaml(path)

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping YAML in {path}")
    return payload


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    """Parse the small YAML subset used by exp1 configs without PyYAML."""
    lines = _yaml_lines(path)
    if not lines:
        return {}
    parsed, next_index = _parse_yaml_block(lines, 0, lines[0][0])
    if next_index != len(lines):
        raise ValueError(f"Could not parse all YAML lines in {path}")
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected mapping YAML in {path}")
    return parsed


def _yaml_lines(path: Path) -> List[tuple[int, str]]:
    out: List[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.split("#", 1)[0].rstrip()
            if not stripped.strip():
                continue
            out.append((len(stripped) - len(stripped.lstrip(" ")), stripped.lstrip()))
    return out


def _parse_yaml_block(
    lines: List[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if lines[index][1].startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_dict(lines, index, indent)


def _parse_yaml_list(
    lines: List[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[List[Any], int]:
    items: List[Any] = []
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent != indent or not text.startswith("- "):
            break
        item_text = text[2:].strip()
        index += 1
        if not item_text:
            value, index = _parse_yaml_block(lines, index, lines[index][0])
            items.append(value)
        elif _looks_like_key_value(item_text):
            key, raw_value = _split_key_value(item_text)
            item: Dict[str, Any] = {}
            if raw_value == "":
                value, index = _parse_yaml_block(lines, index, lines[index][0])
                item[key] = value
            else:
                item[key] = _parse_yaml_scalar(raw_value)
            if index < len(lines) and lines[index][0] > indent:
                extra, index = _parse_yaml_dict(lines, index, lines[index][0])
                item.update(extra)
            items.append(item)
        else:
            items.append(_parse_yaml_scalar(item_text))
    return items, index


def _parse_yaml_dict(
    lines: List[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Dict[str, Any], int]:
    out: Dict[str, Any] = {}
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent or text.startswith("- "):
            break
        if line_indent > indent:
            break
        key, raw_value = _split_key_value(text)
        index += 1
        if raw_value == "":
            if index < len(lines) and lines[index][0] > indent:
                value, index = _parse_yaml_block(lines, index, lines[index][0])
            else:
                value = {}
        else:
            value = _parse_yaml_scalar(raw_value)
        out[key] = value
    return out, index


def _looks_like_key_value(text: str) -> bool:
    return ":" in text and not text.startswith("${")


def _split_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Expected YAML key/value line, got: {text}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_yaml_scalar(text: str) -> Any:
    if text in {"null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        return text[1:-1]
    if text.startswith("[") or text.startswith("{"):
        return ast.literal_eval(text)
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _load_config(path: Optional[Path], project_root: Path) -> Dict[str, Any]:
    if path is None:
        return dict(DEFAULT_CONFIG)

    cfg_path = path if path.is_absolute() else project_root / path
    raw = _load_yaml(cfg_path)
    cfg_dir = cfg_path.parent
    parts: List[Dict[str, Any]] = []
    for entry in raw.get("defaults", []):
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            parts.append(_load_yaml(cfg_dir / f"{entry}.yaml"))
        elif isinstance(entry, dict):
            for group, name in entry.items():
                if name in {None, "null"}:
                    continue
                parts.append(_load_yaml(cfg_dir / str(group) / f"{name}.yaml"))
        else:
            raise TypeError(f"Unsupported defaults entry in {cfg_path}: {entry!r}")
    parts.append(raw)

    merged = dict(DEFAULT_CONFIG)
    for part in parts:
        merged = _recursive_update(merged, part)
    return merged


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--record-json", type=str, default=None)
    parser.add_argument("--record-file", type=Path, default=None)
    parser.add_argument("--chunk", type=Path, default=None, help="JSONL render rows")
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _require_bpy() -> None:
    if bpy is None:
        raise RuntimeError(
            "scripts/render_blender.py must be run with Blender, e.g. "
            "blender --background --python scripts/render_blender.py -- "
            "--chunk rows.jsonl"
        )


def _resolve(project_root: Path, value: Optional[Any]) -> Optional[Path]:
    if value is None or str(value) == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def _load_rows(args: argparse.Namespace, project_root: Path) -> List[Dict[str, Any]]:
    sources = [
        args.record_json is not None,
        args.record_file is not None,
        args.chunk is not None,
    ]
    if sum(sources) != 1:
        raise ValueError(
            "Specify exactly one of --record-json, --record-file, or --chunk"
        )

    if args.record_json is not None:
        row = json.loads(args.record_json)
        if not isinstance(row, dict):
            raise TypeError("--record-json must decode to an object")
        return [row]

    if args.record_file is not None:
        path = _resolve(project_root, args.record_file)
        assert path is not None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return [payload]
        return list(payload)

    chunk = _resolve(project_root, args.chunk)
    assert chunk is not None
    rows: List[Dict[str, Any]] = []
    with chunk.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _default_status_output(
    args: argparse.Namespace,
    project_root: Path,
) -> Optional[Path]:
    if args.status_output is not None:
        return _resolve(project_root, args.status_output)
    if args.chunk is not None:
        chunk = _resolve(project_root, args.chunk)
        assert chunk is not None
        return chunk.with_suffix(".render_status.jsonl")
    return None


def configure_render(cfg: Mapping[str, Any]) -> None:
    render_cfg = cfg.get("render", {})
    scene = bpy.context.scene
    scene.render.engine = str(render_cfg.get("engine", "CYCLES"))
    width, height = [int(v) for v in render_cfg.get("resolution", [224, 224])]
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = bool(
        render_cfg.get("transparent_background", False)
    )
    scene.render.image_settings.file_format = str(render_cfg.get("file_format", "PNG"))

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(render_cfg.get("samples", 64))
        scene.cycles.use_adaptive_sampling = True
        if bool(render_cfg.get("use_gpu", True)):
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.get_devices()
                for dev in prefs.devices:
                    dev.use = dev.type != "CPU"
            except Exception:
                pass


def set_world_color(cfg: Mapping[str, Any]) -> None:
    world_cfg = cfg.get("render", {}).get("world", {})
    rgba = tuple(float(v) for v in world_cfg.get("bg_color", [0.05, 0.05, 0.06, 1.0]))
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    bg = nodes.get("Background") or nodes.new("ShaderNodeBackground")
    out = nodes.get("World Output") or nodes.new("ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value = rgba
    if not bg.outputs["Background"].is_linked:
        world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])


def _record_output_paths(
    record: Mapping[str, Any],
    project_root: Path,
) -> Dict[str, Path]:
    render_id = str(record.get("render_id", "render"))
    output_dir = _resolve(project_root, record.get("output_dir"))
    if output_dir is None:
        rgb_path = _resolve(project_root, record.get("rgb_path"))
        if rgb_path is None:
            output_dir = project_root / "data" / "exp1" / "renders" / render_id
        else:
            output_dir = rgb_path.parent

    return {
        "rgb_path": (
            _resolve(project_root, record.get("rgb_path")) or output_dir / "rgb.png"
        ),
        "depth_path": (
            _resolve(project_root, record.get("depth_path"))
            or output_dir / "depth.npy"
        ),
        "normal_path": (
            _resolve(project_root, record.get("normal_path"))
            or output_dir / "normal_camera.npy"
        ),
        "mask_path": (
            _resolve(project_root, record.get("mask_path")) or output_dir / "mask.npy"
        ),
        "metadata_path": (
            _resolve(project_root, record.get("metadata_path"))
            or output_dir / "render_meta.json"
        ),
    }


def _mesh_path(record: Mapping[str, Any], project_root: Path) -> Path:
    mesh = (
        record.get("normalized_mesh_path")
        or record.get("raw_mesh_path")
        or record.get("mesh_path")
    )
    resolved = _resolve(project_root, mesh)
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"Missing mesh path for render row: {mesh}")
    return resolved


def _mesh_objects() -> List[Any]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def _prepare_scene(
    record: Mapping[str, Any],
    cfg: Mapping[str, Any],
    project_root: Path,
):
    from mesh_utils import center_and_normalize, import_mesh

    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_render(cfg)
    set_world_color(cfg)

    obj = import_mesh(str(_mesh_path(record, project_root)))
    center_and_normalize(obj)
    scale = float(record.get("object_scale", 1.0))
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    objects = _mesh_objects()
    material_meta = apply_materials(objects, record, cfg)
    cam_obj = setup_camera(record, cfg)
    light_meta = setup_lighting(record, cfg)
    bpy.context.view_layer.update()
    return cam_obj, light_meta, material_meta


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_record(
    record: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    start = time.time()
    output_paths = _record_output_paths(record, project_root)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    cam_obj, light_meta, material_meta = _prepare_scene(record, cfg, project_root)
    _write_rgb(output_paths["rgb_path"])

    scene = bpy.context.scene
    width = int(scene.render.resolution_x)
    height = int(scene.render.resolution_y)
    buffers = render_geometry_buffers(
        scene=scene,
        cam_obj=cam_obj,
        width=width,
        height=height,
    )
    np.save(output_paths["depth_path"], buffers["depth"])
    np.save(output_paths["normal_path"], buffers["normal_camera"])
    np.save(output_paths["mask_path"], buffers["mask"])

    meta = {
        **dict(record),
        "rgb_path": str(output_paths["rgb_path"]),
        "depth_path": str(output_paths["depth_path"]),
        "normal_path": str(output_paths["normal_path"]),
        "mask_path": str(output_paths["mask_path"]),
        "render_status": "success",
        "qc_error_message": "",
        "normal_coordinate_frame": "camera",
        "depth_convention": "positive camera-space z distance; NaN background",
        "mask_convention": "boolean foreground mask from ray casting",
        "blender_version": bpy.app.version_string,
        "render_elapsed_sec": time.time() - start,
        "buffer_shapes": buffer_shapes(buffers),
        **material_meta,
        **camera_metadata(cam_obj),
        **light_meta,
    }
    with output_paths["metadata_path"].open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


def _failure_row(record: Mapping[str, Any], exc: BaseException) -> Dict[str, Any]:
    row = dict(record)
    row["render_status"] = "failed"
    row["qc_error_message"] = f"{type(exc).__name__}: {exc}"
    row["traceback"] = traceback.format_exc()
    return row


def _write_status(path: Optional[Path], rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(row)) for row in rows]
    if path is None:
        for line in lines:
            print(line)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = _parse_args(argv)
    _require_bpy()
    project_root = args.project_root.expanduser().resolve()
    _ensure_paths(project_root)
    cfg = _load_config(args.config, project_root)
    rows = _load_rows(args, project_root)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    fail_fast = bool(args.fail_fast or cfg.get("render", {}).get("fail_fast", False))

    status_rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(rows):
        render_id = record.get("render_id", f"row_{idx}")
        print(f"[{idx + 1}/{len(rows)}] Rendering {render_id}")
        try:
            status_rows.append(
                render_record(record, cfg=cfg, project_root=project_root)
            )
        except Exception as exc:
            failure = _failure_row(record, exc)
            status_rows.append(failure)
            print(f"FAILED {render_id}: {failure['qc_error_message']}", file=sys.stderr)
            if fail_fast:
                _write_status(_default_status_output(args, project_root), status_rows)
                raise
        finally:
            from mesh_utils import clear_meshes

            clear_meshes()

    status_output = _default_status_output(args, project_root)
    _write_status(status_output, status_rows)
    if status_output is not None:
        print(f"Wrote render status rows to {status_output}")


if __name__ == "__main__":
    main()

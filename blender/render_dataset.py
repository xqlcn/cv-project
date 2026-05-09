"""
Blender batch renderer for chirality pairs (+ optional viewpoint jitter).

Launch (from project root):

  blender --background --python blender/render_dataset.py -- \\
      --config configs/render_config.yaml \\
      --project-root .

The mesh manifest is JSON: list of {object_id, category, mesh_path}. Meshes may be
ModelNet .off files or OBJ/GLB/FBX (see blender/mesh_utils.py).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import bpy


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _ensure_blender_path(script_dir: Path) -> None:
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


_ensure_blender_path(_script_dir())

from camera_utils import ensure_camera, set_camera_extrinsics, set_camera_intrinsics  # noqa: E402
from lighting_utils import add_area_fill, add_sun, clear_lights  # noqa: E402
from material_utils import assign_principled_material  # noqa: E402
from mesh_utils import (  # noqa: E402
    center_and_normalize,
    clear_meshes,
    delete_object,
    duplicate_mesh_mirror,
    import_mesh,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required inside Blender's Python. "
            "Example: /Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11 "
            "-m pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render chirality dataset in Blender")
    parser.add_argument("--config", type=str, required=True, help="Path to render_config.yaml")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root for resolving relative paths in config",
    )
    return parser.parse_args(argv)


def _resolve(root: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (root / p)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def configure_render(
    *,
    resolution: List[int],
    engine: str,
    samples: int,
    use_gpu: bool,
    transparent: bool,
    file_format: str,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.image_settings.file_format = file_format
    scene.render.film_transparent = transparent

    if engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_adaptive_sampling = True
        if use_gpu:
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.get_devices()
                for dev in prefs.devices:
                    dev.use = dev.type != "CPU"
            except Exception:
                pass
    elif engine == "BLENDER_EEVEE":
        # EEVEE uses different sampling settings; keep defaults for now
        pass


def set_world_color(rgba: List[float]) -> None:
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = tuple(rgba)
    out = world.node_tree.nodes.get("World Output") or world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])


def _sun_direction_vector(elevation_deg: float, azimuth_deg: float) -> List[float]:
    import math

    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    return [
        float(math.cos(el) * math.cos(az)),
        float(math.cos(el) * math.sin(az)),
        float(math.sin(el)),
    ]


def render_object_views(
    *,
    cfg: Dict[str, Any],
    entry: Dict[str, Any],
    project_root: Path,
    rng: random.Random,
    setup_lights: bool,
) -> List[Dict[str, Any]]:
    mesh_path = Path(entry["mesh_path"]).expanduser()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing mesh: {mesh_path}")

    tex_type = str(cfg["texture"]["default_type"])
    mirror_axis = str(cfg["chirality"]["mirror_axis"])
    jitter = float(cfg["chirality"]["view_jitter_rad"])
    viewpoints = cfg["chirality"]["viewpoints"]

    out_dir = _resolve(project_root, cfg["paths"]["output_images"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: List[Dict[str, Any]] = []

    base = import_mesh(str(mesh_path))
    base.name = f"{entry['object_id']}_base"
    center_and_normalize(base)
    assign_principled_material(base, texture_type=tex_type, seed=rng.randint(0, 10_000))

    mirrored = duplicate_mesh_mirror(base, axis=mirror_axis)
    mirrored.name = f"{entry['object_id']}_mirrored"
    assign_principled_material(mirrored, texture_type=tex_type, seed=rng.randint(0, 10_000))

    cam_obj = ensure_camera()
    set_camera_intrinsics(
        cam_obj.data,
        fov_deg=float(cfg["camera"]["fov_deg"]),
        clip_start=float(cfg["camera"]["clip_start"]),
        clip_end=float(cfg["camera"]["clip_end"]),
    )

    sun_el, sun_az = [float(x) for x in cfg["lighting"]["sun_angles_deg"]]
    light_vec = _sun_direction_vector(sun_el, sun_az)

    if setup_lights:
        clear_lights()
        add_sun("KeySun", elevation_deg=sun_el, azimuth_deg=sun_az, energy=float(cfg["lighting"]["sun_energy"]))
        add_area_fill(
            "FillArea",
            location=(2.0, -2.5, 3.0),
            energy=float(cfg["lighting"]["fill_energy"]),
        )

    for vi, vp in enumerate(viewpoints):
        az = float(vp["azimuth"])
        el = float(vp["elevation"])
        dist = float(vp["distance"])
        pair_id = f"{entry['object_id']}_v{vi}"

        def render_active(obj: bpy.types.Object, suffix: str, meta_extra: Dict[str, Any]) -> None:
            base.hide_render = obj != base
            mirrored.hide_render = obj != mirrored
            base.hide_viewport = base.hide_render
            mirrored.hide_viewport = mirrored.hide_render

            fname = f"{entry['object_id']}_{suffix}.png"
            fpath = out_dir / fname
            bpy.context.scene.render.filepath = str(fpath)
            bpy.ops.render.render(write_still=True)

            row = {
                "object_id": entry["object_id"],
                "category": entry.get("category", "unknown"),
                "chirality_label": meta_extra.get("chirality_label", -1),
                "sample_kind": meta_extra.get("sample_kind", "unknown"),
                "pair_group_id": pair_id,
                "camera_distance": dist,
                "azimuth": az,
                "elevation": el,
                "lighting_direction": light_vec,
                "texture_type": tex_type,
                "render_path": str(fpath.resolve()),
            }
            meta_rows.append(row)

        set_camera_extrinsics(cam_obj, azimuth=az, elevation=el, distance=dist)
        render_active(
            base,
            suffix=f"v{vi}_orig",
            meta_extra={"chirality_label": 0, "sample_kind": "chirality_original"},
        )
        render_active(
            mirrored,
            suffix=f"v{vi}_mirror",
            meta_extra={"chirality_label": 1, "sample_kind": "chirality_mirror"},
        )

        # Viewpoint jitter (original mesh only) for cosine-similarity baseline
        az_j = az + jitter
        set_camera_extrinsics(cam_obj, azimuth=az_j, elevation=el, distance=dist)
        render_active(
            base,
            suffix=f"v{vi}_viewjit",
            meta_extra={"chirality_label": -1, "sample_kind": "view_jitter"},
        )

    delete_object(mirrored)
    delete_object(base)
    return meta_rows


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = _parse_args(argv)
    project_root = Path(args.project_root).resolve()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = project_root / args.config
    cfg = _load_yaml(cfg_path)

    manifest_path = _resolve(project_root, cfg["paths"]["mesh_manifest"])
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest: List[Dict[str, Any]] = json.load(f)

    limits = cfg.get("limits", {})
    max_objects = limits.get("max_objects")
    seed = int(limits.get("seed", 0))
    rng = random.Random(seed)

    reset_scene()
    configure_render(
        resolution=list(cfg["render"]["resolution"]),
        engine=str(cfg["render"]["engine"]),
        samples=int(cfg["render"]["samples"]),
        use_gpu=bool(cfg["render"]["use_gpu"]),
        transparent=bool(cfg["render"]["transparent_background"]),
        file_format=str(cfg["render"]["file_format"]),
    )
    set_world_color(list(cfg["world"]["bg_color"]))

    if max_objects is not None:
        manifest = manifest[: int(max_objects)]

    all_meta: List[Dict[str, Any]] = []
    for i, entry in enumerate(manifest):
        print(f"[{i+1}/{len(manifest)}] Rendering {entry.get('object_id')}")
        rows = render_object_views(
            cfg=cfg,
            entry=entry,
            project_root=project_root,
            rng=rng,
            setup_lights=(i == 0),
        )
        all_meta.extend(rows)

    meta_out = _resolve(project_root, cfg["paths"]["output_metadata"])
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with meta_out.open("w", encoding="utf-8") as f:
        for row in all_meta:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(all_meta)} metadata rows to {meta_out}")


if __name__ == "__main__":
    main()

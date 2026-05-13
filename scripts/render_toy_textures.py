#!/usr/bin/env python3
"""Render a tiny set of texture-condition examples for visual inspection.

This is a quick sanity-check renderer that does NOT depend on having a full
asset/manifest pipeline already built. It hard-codes a handful of ShapeNet
test objects, generates a small JSONL render plan with one row per
(object, texture_condition), and shells out to Blender via the standard
``scripts/render_blender.py`` entrypoint.

Outputs land under ``outputs/toy_renders/<render_id>/`` (rgb.png + geometry
buffers + render_meta.json). After it finishes, the script also builds a
simple contact sheet at ``outputs/toy_renders/contact_sheet.png`` so the
three texture conditions can be compared side-by-side per object.

Run from the repo root:

    python scripts/render_toy_textures.py
    # Or with a specific Blender binary:
    BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender \
        python scripts/render_toy_textures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TOY_OBJECTS: List[dict] = [
    {
        "object_id": "airplane_1021a0914a7207aff927ed529ad90a11",
        "category": "airplane",
        "source_dataset": "shapenet",
        "raw_mesh_path": (
            "data/shapenet_hf/extracted/02691156/02691156/"
            "1021a0914a7207aff927ed529ad90a11/models/model_normalized.obj"
        ),
        "normalized_mesh_path": (
            "data/exp1_under12h/normalized_assets/shapenet/test/airplane/"
            "airplane_1021a0914a7207aff927ed529ad90a11.glb"
        ),
        "has_photorealistic_material": True,
    },
    {
        "object_id": "bench_117259de2f72887bad5067eac75a07f7",
        "category": "bench",
        "source_dataset": "shapenet",
        "raw_mesh_path": (
            "data/shapenet_hf/extracted/02828884/02828884/"
            "117259de2f72887bad5067eac75a07f7/models/model_normalized.obj"
        ),
        "normalized_mesh_path": (
            "data/exp1_under12h/normalized_assets/shapenet/test/bench/"
            "bench_117259de2f72887bad5067eac75a07f7.glb"
        ),
        "has_photorealistic_material": True,
    },
    {
        "object_id": "chair_103c31671f8c0b1467bb14b25f99796e",
        "category": "chair",
        "source_dataset": "shapenet",
        "raw_mesh_path": (
            "data/shapenet_hf/extracted/03001627/03001627/"
            "103c31671f8c0b1467bb14b25f99796e/models/model_normalized.obj"
        ),
        "normalized_mesh_path": (
            "data/exp1_under12h/normalized_assets/shapenet/test/chair/"
            "chair_103c31671f8c0b1467bb14b25f99796e.glb"
        ),
        "has_photorealistic_material": True,
    },
    {
        "object_id": "car_12243301d1c8148e33d7c9e122eec9b6",
        "category": "car",
        "source_dataset": "shapenet",
        "raw_mesh_path": (
            "data/shapenet_hf/extracted/02958343/02958343/"
            "12243301d1c8148e33d7c9e122eec9b6/models/model_normalized.obj"
        ),
        "normalized_mesh_path": (
            "data/exp1_under12h/normalized_assets/shapenet/test/car/"
            "car_12243301d1c8148e33d7c9e122eec9b6.glb"
        ),
        "has_photorealistic_material": True,
    },
]

TEXTURE_CONDITIONS: Sequence[str] = ("photorealistic", "flat", "random_noise")


def _render_id(parts: Sequence[str]) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:14]
    return f"toy_{digest}"


def _build_plan(
    output_root: Path,
    *,
    camera_distance: float,
    camera_azimuth_deg: float,
    camera_elevation_deg: float,
    camera_fov_deg: float,
    light_azimuth_deg: float,
    light_elevation_deg: float,
    light_intensity: float,
    object_scale: float,
) -> List[dict]:
    plan: List[dict] = []
    for record in TOY_OBJECTS:
        for tex in TEXTURE_CONDITIONS:
            object_id = str(record["object_id"])
            render_id = _render_id([object_id, tex])
            out_dir = output_root / render_id
            row = {
                "render_id": render_id,
                "object_id": object_id,
                "category": record["category"],
                "source_dataset": record["source_dataset"],
                "split": "test",
                "raw_mesh_path": str(PROJECT_ROOT / record["raw_mesh_path"]),
                "normalized_mesh_path": str(
                    PROJECT_ROOT / record["normalized_mesh_path"]
                ),
                "has_photorealistic_material": bool(
                    record.get("has_photorealistic_material", True)
                ),
                "texture_condition": tex,
                "texture_seed": int.from_bytes(
                    hashlib.sha1(f"{object_id}|{tex}".encode("utf-8")).digest()[:4],
                    "big",
                ),
                "camera_distance": float(camera_distance),
                "camera_azimuth_deg": float(camera_azimuth_deg),
                "camera_elevation_deg": float(camera_elevation_deg),
                "camera_fov_deg": float(camera_fov_deg),
                "object_scale": float(object_scale),
                "light_type": "sun",
                "light_azimuth_deg": float(light_azimuth_deg),
                "light_elevation_deg": float(light_elevation_deg),
                "light_intensity": float(light_intensity),
                "render_seed": 0,
                "render_status": "pending",
                "qc_error_message": "",
                "rgb_path": str(out_dir / "rgb.png"),
                "depth_path": str(out_dir / "depth.npy"),
                "normal_path": str(out_dir / "normal_camera.npy"),
                "mask_path": str(out_dir / "mask.npy"),
            }
            plan.append(row)
    return plan


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row)) + "\n")


def _resolve_blender_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("BLENDER_BIN")
    if env:
        return env
    which = shutil.which("blender")
    if which:
        return which
    mac_default = "/Applications/Blender.app/Contents/MacOS/Blender"
    if Path(mac_default).is_file():
        return mac_default
    raise SystemExit(
        "Could not find blender. Pass --blender-bin or set BLENDER_BIN."
    )


def _make_contact_sheet(plan: Sequence[Mapping[str, object]], out_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        print(f"Skipping contact sheet (Pillow unavailable): out={out_path}")
        return

    rows: List[dict] = []
    for row in plan:
        rgb_path = Path(str(row["rgb_path"]))
        if rgb_path.is_file():
            rows.append(dict(row))
    if not rows:
        return

    object_ids: List[str] = []
    for row in rows:
        oid = str(row["object_id"])
        if oid not in object_ids:
            object_ids.append(oid)
    textures = list(TEXTURE_CONDITIONS)

    tile = 224
    label_h = 24
    pad = 6
    cols = len(textures)
    rows_count = len(object_ids)
    width = pad + cols * (tile + pad)
    height = pad + label_h + rows_count * (tile + label_h + pad)

    sheet = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for ci, tex in enumerate(textures):
        x = pad + ci * (tile + pad)
        draw.text((x + 4, 4), tex, fill=(20, 20, 20), font=font)

    by_key = {
        (str(row["object_id"]), str(row["texture_condition"])): row for row in rows
    }
    for ri, oid in enumerate(object_ids):
        y_label = pad + label_h + ri * (tile + label_h + pad)
        draw.text(
            (pad + 4, y_label - label_h + 4),
            oid.split("_", 1)[0],
            fill=(20, 20, 20),
            font=font,
        )
        for ci, tex in enumerate(textures):
            row = by_key.get((oid, tex))
            if row is None:
                continue
            tile_img = Image.open(str(row["rgb_path"])).convert("RGB")
            tile_img.thumbnail((tile, tile))
            x = pad + ci * (tile + pad)
            sheet.paste(tile_img, (x + (tile - tile_img.width) // 2, y_label))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"Wrote contact sheet to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "toy_renders",
        help="Directory for per-render output and the contact sheet.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "exp1_under12h.yaml",
        help="Hydra config to feed to scripts/render_blender.py.",
    )
    parser.add_argument("--blender-bin", type=str, default=None)
    parser.add_argument("--camera-distance", type=float, default=1.8)
    parser.add_argument("--camera-azimuth-deg", type=float, default=35.0)
    parser.add_argument("--camera-elevation-deg", type=float, default=20.0)
    parser.add_argument("--camera-fov-deg", type=float, default=50.0)
    parser.add_argument("--light-azimuth-deg", type=float, default=55.0)
    parser.add_argument("--light-elevation-deg", type=float, default=40.0)
    parser.add_argument("--light-intensity", type=float, default=3.0)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only build the JSONL plan + contact sheet from existing renders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    plan_path = output_root / "render_plan.jsonl"
    status_path = output_root / "render_status.jsonl"
    contact_sheet = output_root / "contact_sheet.png"

    plan = _build_plan(
        output_root,
        camera_distance=args.camera_distance,
        camera_azimuth_deg=args.camera_azimuth_deg,
        camera_elevation_deg=args.camera_elevation_deg,
        camera_fov_deg=args.camera_fov_deg,
        light_azimuth_deg=args.light_azimuth_deg,
        light_elevation_deg=args.light_elevation_deg,
        light_intensity=args.light_intensity,
        object_scale=args.object_scale,
    )

    for row in plan:
        for required in ("raw_mesh_path", "normalized_mesh_path"):
            path = Path(str(row[required]))
            if not path.is_file():
                print(f"WARNING: missing {required}={path}", file=sys.stderr)

    _write_jsonl(plan_path, plan)
    print(f"Wrote toy render plan ({len(plan)} rows) to {plan_path}")

    if not args.skip_render:
        blender_bin = _resolve_blender_bin(args.blender_bin)
        cmd = [
            blender_bin,
            "--background",
            "--python",
            str(PROJECT_ROOT / "scripts" / "render_blender.py"),
            "--",
            "--config",
            str(args.config),
            "--project-root",
            str(PROJECT_ROOT),
            "--chunk",
            str(plan_path),
            "--status-output",
            str(status_path),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    _make_contact_sheet(plan, contact_sheet)


if __name__ == "__main__":
    main()

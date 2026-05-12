#!/usr/bin/env python3
"""
ModelNet40 preprocessing + rendering pipeline (Steps 1-8).

Steps implemented:
1) Scan ModelNet40 folders (sanity)
2) Load each .off mesh
3) Triangulate faces
4) Center mesh at origin
5) Scale to unit size
6) Compute/fix normals
7) Render controlled multi-view images (default 16 views)
8) Save RGB/depth/normal + metadata JSONL
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow `python scripts/preprocess_modelnet40.py` from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.modelnet40_index import discover_modelnet40_records
from src.rendering.mesh_renderer import MeshMultiviewRenderer, RenderConfig
from src.utils.io import ensure_dir


def _triangulate_mesh(mesh: Any) -> Any:
    """Ensure triangle faces when possible."""
    import trimesh

    m = mesh.copy()
    if hasattr(m, "triangulate"):
        try:
            tri = m.triangulate()
            if tri is not None:
                m = tri
        except Exception:
            pass

    if getattr(m.faces, "ndim", 2) == 2 and m.faces.shape[1] == 3:
        return m
    if getattr(m.faces, "ndim", 2) == 2 and m.faces.shape[1] == 4:
        faces = trimesh.geometry.triangulate_quads(m.faces)
        return trimesh.Trimesh(vertices=m.vertices.copy(), faces=faces, process=False)
    return m


def _center_scale_fix_normals(mesh: Any) -> Any:
    """Center to origin, scale to unit extent, and fix normals."""
    m = mesh.copy()
    if len(m.vertices) == 0:
        return m
    if hasattr(m, "remove_unreferenced_vertices"):
        m.remove_unreferenced_vertices()

    # Trimesh API varies by version; use whichever cleanup methods exist.
    if hasattr(m, "remove_degenerate_faces"):
        m.remove_degenerate_faces()
    elif hasattr(m, "nondegenerate_faces") and hasattr(m, "update_faces"):
        try:
            m.update_faces(m.nondegenerate_faces())
        except Exception:
            pass

    if hasattr(m, "remove_duplicate_faces"):
        m.remove_duplicate_faces()
    elif hasattr(m, "unique_faces") and hasattr(m, "update_faces"):
        try:
            m.update_faces(m.unique_faces())
        except Exception:
            pass

    try:
        m.process(validate=True)
    except TypeError:
        m.process()

    if hasattr(m, "fix_normals"):
        m.fix_normals()

    verts = np.asarray(m.vertices, dtype=np.float64)
    center = verts.mean(axis=0)
    verts = verts - center
    max_extent = float(np.ptp(verts, axis=0).max())
    if max_extent > 1e-8:
        verts = verts / max_extent
    m.vertices = verts
    if hasattr(m, "fix_normals"):
        m.fix_normals()
    return m


def _load_preprocess_mesh(path: Path) -> Any:
    import trimesh

    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError(f"No mesh geometry in scene: {path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh for {path}, got {type(loaded)}")

    loaded = _triangulate_mesh(loaded)
    loaded = _center_scale_fix_normals(loaded)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ModelNet40 and render multiview data.")
    parser.add_argument("--root", type=Path, default=Path("data/modelnet40"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/modelnet40"))
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--max-meshes", type=int, default=100, help="Sanity default; set null-equivalent with -1")
    parser.add_argument("--n-views", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = ("train", "test") if args.split == "both" else (args.split,)
    records = discover_modelnet40_records(args.root, splits=splits)
    if not records:
        raise SystemExit(f"No ModelNet records found under {args.root.resolve()}")
    if args.max_meshes >= 0:
        records = records[: args.max_meshes]

    print(f"[Step 1] scan ok: {len(records)} meshes from splits={splits}")

    out_root = args.output_root.resolve()
    rgb_root = ensure_dir(out_root / "images")
    depth_root = ensure_dir(out_root / "depth")
    normal_root = ensure_dir(out_root / "normal")
    mesh_root = ensure_dir(out_root / "meshes")
    meta_root = ensure_dir(out_root / "metadata")
    meta_path = meta_root / "modelnet40_views.jsonl"

    categories = sorted({r["category"] for r in records})
    cat_to_id = {c: i for i, c in enumerate(categories)}
    (meta_root / "category_to_id.json").write_text(json.dumps(cat_to_id, indent=2) + "\n", encoding="utf-8")

    renderer = MeshMultiviewRenderer(
        RenderConfig(
            image_size=(args.image_size, args.image_size),
            n_views=args.n_views,
            camera_mode="fibonacci",
            seed=args.seed,
        )
    )

    rows_out: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc="preprocess+render"):
        mesh_path = Path(rec["mesh_path"])
        category = rec["category"]
        split = rec["split"]
        object_id = rec["object_id"]

        # Steps 2-6
        mesh = _load_preprocess_mesh(mesh_path)

        # Save processed mesh for reproducibility
        proc_mesh_dir = ensure_dir(mesh_root / split / category)
        proc_mesh_path = proc_mesh_dir / f"{object_id}.obj"
        mesh.export(proc_mesh_path)

        # Step 7
        views = renderer.render_mesh(mesh, mesh_path_for_meta=str(proc_mesh_path))

        # Step 8
        for v in views:
            vid = int(v["view_id"])
            stem = f"{object_id}_v{vid:03d}"

            rgb_dir = ensure_dir(rgb_root / split / category)
            depth_dir = ensure_dir(depth_root / split / category)
            normal_dir = ensure_dir(normal_root / split / category)

            rgb_path = rgb_dir / f"{stem}.png"
            depth_path = depth_dir / f"{stem}.npy"
            normal_path = normal_dir / f"{stem}.npy"
            depth_vis_path = depth_dir / f"{stem}.png"
            normal_vis_path = normal_dir / f"{stem}.png"

            Image.fromarray(v["rgb"]).save(rgb_path)
            np.save(depth_path, v["depth"])
            np.save(normal_path, v["normal"])

            # Save CLIP-friendly visualizations for non-RGB modalities.
            depth = np.asarray(v["depth"], dtype=np.float32)
            fg = np.isfinite(depth)
            if fg.any():
                dmin = float(depth[fg].min())
                dmax = float(depth[fg].max())
                dnorm = (depth - dmin) / max(1e-8, (dmax - dmin))
            else:
                dnorm = np.zeros_like(depth, dtype=np.float32)
            d8 = (np.clip(dnorm, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(d8, mode="L").save(depth_vis_path)

            n = np.asarray(v["normal"], dtype=np.float32)
            nvis = ((np.clip(n, -1.0, 1.0) * 0.5 + 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(nvis, mode="RGB").save(normal_vis_path)

            pose = np.asarray(v["camera_pose"], dtype=np.float32)
            eye = pose[:3, 3]
            r = float(np.linalg.norm(eye) + 1e-12)
            azimuth = float(math.atan2(float(eye[2]), float(eye[0])))
            elevation = float(math.asin(float(np.clip(eye[1] / r, -1.0, 1.0))))

            rows_out.append(
                {
                    "mesh_path": str(mesh_path.resolve()),
                    "processed_mesh_path": str(proc_mesh_path.resolve()),
                    "category": category,
                    "category_id": cat_to_id[category],
                    "split": split,
                    "object_id": object_id,
                    "view_id": vid,
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "render_path": str(rgb_path.resolve()),
                    "depth_path": str(depth_path.resolve()),
                    "normal_path": str(normal_path.resolve()),
                    "depth_vis_path": str(depth_vis_path.resolve()),
                    "normal_vis_path": str(normal_vis_path.resolve()),
                }
            )

    with meta_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    print(f"[Step 8] wrote {len(rows_out)} view rows -> {meta_path}")
    print("[Step 9] next: python -m src.training.extract_features --config-name=modelnet_clip")
    print("[Step 10] next: python -m src.training.train_modelnet_probe --config-name=modelnet_clip")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Backfill azimuth/elevation and CLIP-friendly depth/normal visualizations for an existing
`data/processed/modelnet40/metadata/modelnet40_views.jsonl` produced before these fields existed.
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

# Allow direct script execution from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rendering.camera_utils import fibonacci_sphere_views


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=Path("data/processed/modelnet40/metadata/modelnet40_views.jsonl"))
    p.add_argument("--n-views", type=int, default=16)
    p.add_argument("--distance", type=float, default=2.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    meta = args.metadata.resolve()
    rows = _load_jsonl(meta)
    if not rows:
        raise SystemExit(f"No rows found in {meta}")

    poses = fibonacci_sphere_views(args.n_views, radius=args.distance, seed=args.seed)
    changed = 0
    for row in rows:
        vid = int(row.get("view_id", 0))
        vid = max(0, min(args.n_views - 1, vid))
        eye = poses[vid][:3, 3]
        r = float(np.linalg.norm(eye) + 1e-12)
        row["azimuth"] = float(math.atan2(float(eye[2]), float(eye[0])))
        row["elevation"] = float(math.asin(float(np.clip(eye[1] / r, -1.0, 1.0))))

        # depth visualization
        if "depth_vis_path" not in row and "depth_path" in row:
            dpath = Path(row["depth_path"])
            if dpath.is_file():
                depth = np.load(dpath)
                fg = np.isfinite(depth)
                if fg.any():
                    dmin = float(depth[fg].min())
                    dmax = float(depth[fg].max())
                    dnorm = (depth - dmin) / max(1e-8, (dmax - dmin))
                else:
                    dnorm = np.zeros_like(depth, dtype=np.float32)
                d8 = (np.clip(dnorm, 0.0, 1.0) * 255.0).astype(np.uint8)
                vis = dpath.with_suffix(".png")
                Image.fromarray(d8, mode="L").save(vis)
                row["depth_vis_path"] = str(vis.resolve())

        # normal visualization
        if "normal_vis_path" not in row and "normal_path" in row:
            npath = Path(row["normal_path"])
            if npath.is_file():
                n = np.load(npath)
                nvis = ((np.clip(n, -1.0, 1.0) * 0.5 + 0.5) * 255.0).astype(np.uint8)
                vis = npath.with_suffix(".png")
                Image.fromarray(nvis, mode="RGB").save(vis)
                row["normal_vis_path"] = str(vis.resolve())

        changed += 1

    _write_jsonl(meta, rows)
    print(f"Updated {changed} rows in {meta}")


if __name__ == "__main__":
    main()

"""
Synthetic primitive meshes for **controlled** experiments (default pipeline).

Meshes are written under ``data/synthetic_primitives/{split}/`` as ``.obj`` files so
they consume the same trimesh/pyrender path as imported meshes. Records match the
shared schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.rendered_mesh_dataset import RenderedMeshDataset
from src.rendering.mesh_renderer import RenderConfig


def default_synthetic_root(project_root: Optional[Union[str, Path]] = None) -> Path:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    return root / "data" / "synthetic_primitives"


_PRIMITIVE_BUILDERS = ("box", "sphere", "cylinder", "cone", "capsule")


def _make_primitive(kind: str, rng: np.random.Generator) -> "Any":
    import trimesh

    kind = kind.lower()
    if kind == "box":
        ext = rng.uniform(0.35, 0.9, size=3)
        return trimesh.creation.box(extents=ext)
    if kind == "sphere":
        r = float(rng.uniform(0.4, 0.55))
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    if kind == "cylinder":
        r, h = float(rng.uniform(0.25, 0.45)), float(rng.uniform(0.5, 1.0))
        return trimesh.creation.cylinder(radius=r, height=h, sections=32)
    if kind == "cone":
        r, h = float(rng.uniform(0.3, 0.5)), float(rng.uniform(0.55, 1.0))
        return trimesh.creation.cone(radius=r, height=h, sections=32)
    if kind == "capsule":
        r, h = float(rng.uniform(0.2, 0.35)), float(rng.uniform(0.6, 1.0))
        return trimesh.creation.capsule(radius=r, height=h, count=[16, 16])
    raise ValueError(f"Unknown primitive kind: {kind}")


def build_synthetic_primitive_records(
    *,
    root: Union[str, Path],
    n_train: int = 200,
    n_val: int = 40,
    seed: int = 0,
    kinds: Sequence[str] = _PRIMITIVE_BUILDERS,
) -> List[Dict[str, Any]]:
    """
    Create (or refresh) ``.obj`` meshes on disk and return record dicts.

    Splits: ``train`` / ``val`` (stored under ``root/train`` and ``root/val``).
    """
    root = Path(root).expanduser().resolve()
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    def emit(split: str, n: int, offset: int) -> None:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            kind = str(kinds[(offset + i) % len(kinds)])
            mesh = _make_primitive(kind, rng)
            mesh.vertices += rng.normal(scale=0.02, size=mesh.vertices.shape)
            oid = f"syn_{split}_{kind}_{offset + i:05d}"
            fname = f"{oid}.obj"
            fpath = split_dir / fname
            mesh.export(fpath)
            records.append(
                {
                    "mesh_path": str(fpath.resolve()),
                    "category": kind,
                    "split": split,
                    "object_id": oid,
                    "dataset": "synthetic_primitives",
                }
            )

    emit("train", n_train, 0)
    emit("val", n_val, n_train)
    meta_path = root / "catalog.json"
    meta_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def load_synthetic_catalog(root: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(root) / "catalog.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run build_synthetic_primitive_records(...) or "
            "scripts/setup_synthetic_primitives.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


class SyntheticPrimitiveMeshDataset(Dataset):
    """Mesh-only index for cached synthetic primitives."""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        split: Literal["train", "val", "both"] = "train",
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_synthetic_root(project_root)
        all_recs = load_synthetic_catalog(self.root)
        if split == "both":
            self.records = all_recs
        else:
            self.records = [r for r in all_recs if r["split"] == split]
        cats = sorted({r["category"] for r in self.records})
        self._cat_to_idx = {c: i for i, c in enumerate(cats)}
        self.categories = cats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        r = self.records[index]
        return {
            "mesh_path": r["mesh_path"],
            "category": r["category"],
            "split": r["split"],
            "object_id": r["object_id"],
            "category_id": self._cat_to_idx[r["category"]],
            "dataset": r["dataset"],
        }


class RenderedSyntheticPrimitiveDataset(RenderedMeshDataset):
    """Default rendered pipeline on synthetic primitives."""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        split: Literal["train", "val", "both"] = "train",
        render_cfg: Optional[RenderConfig] = None,
        flatten_views: bool = True,
        project_root: Optional[Union[str, Path]] = None,
        max_cache_objects: int = 64,
    ) -> None:
        self.root = Path(root) if root is not None else default_synthetic_root(project_root)
        records = load_synthetic_catalog(self.root)
        if split != "both":
            records = [r for r in records if r["split"] == split]
        super().__init__(
            records,
            render_cfg=render_cfg,
            flatten_views=flatten_views,
            max_cache_objects=max_cache_objects,
        )

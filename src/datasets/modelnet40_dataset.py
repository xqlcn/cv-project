"""
ModelNet40 mesh index and rendered PyTorch datasets.

Expected layout (normalized)::

    data/modelnet40/
      train/<category>/*.off
      test/<category>/*.off

The official Princeton ZIP ships ``ModelNet40/<category>/train|test/*.off``; run
``python scripts/normalize_modelnet40_layout.py`` once to reorder (the download
script does this automatically after unzip).

The same record schema is used for other mesh corpora (e.g. future ShapeNet) so
``RenderedMeshDataset`` stays agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from src.datasets.rendered_mesh_dataset import RenderedMeshDataset
from src.rendering.mesh_renderer import RenderConfig


def default_modelnet40_root(project_root: Optional[Union[str, Path]] = None) -> Path:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    return root / "data" / "modelnet40"


def discover_modelnet40_records(
    root: Union[str, Path],
    *,
    splits: Sequence[str] = ("train", "test"),
) -> List[Dict[str, Any]]:
    """
    Scan ``root/{split}/{category}/*.off`` (normalized layout).

    If only the raw Princeton tree exists (``root/ModelNet40/<category>/{split}/``),
    records are still discovered so you can list data before normalizing.

    Each record:
        mesh_path, category, split, object_id, dataset="modelnet40"
    """
    root = Path(root).expanduser().resolve()
    records: List[Dict[str, Any]] = []

    def add_from_split_category(split: str, category: str, cat_dir: Path) -> None:
        for off in sorted(cat_dir.glob("*.off")):
            stem = off.stem
            oid = f"{category}_{stem}"
            records.append(
                {
                    "mesh_path": str(off.resolve()),
                    "category": category,
                    "split": split,
                    "object_id": oid,
                    "dataset": "modelnet40",
                }
            )

    for split in splits:
        split_dir = root / split
        if split_dir.is_dir():
            for cat_dir in sorted(split_dir.iterdir()):
                if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                    continue
                add_from_split_category(split, cat_dir.name, cat_dir)

    if records:
        records.sort(key=lambda r: (r["split"], r["category"], r["object_id"]))
        return records

    # Raw ZIP layout: ModelNet40/<category>/train|test/
    bundled = root / "ModelNet40"
    if bundled.is_dir():
        for cat_dir in sorted(bundled.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                continue
            category = cat_dir.name
            for split in splits:
                d = cat_dir / split
                if d.is_dir():
                    add_from_split_category(split, category, d)

    records.sort(key=lambda r: (r["split"], r["category"], r["object_id"]))
    return records


def save_records_json(path: Union[str, Path], records: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=2) + "\n", encoding="utf-8")


def load_records_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        return list(data["records"])
    return list(data)


class ModelNet40MeshDataset(Dataset):
    """
    Mesh-only index: ``__getitem__`` returns paths and labels (no images).

    Use for manifests or custom rendering; for RGB/depth/normal use
    :class:`RenderedModelNetDataset`.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        split: Literal["train", "test", "both"] = "train",
        categories: Optional[Sequence[str]] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        root = Path(root) if root is not None else default_modelnet40_root(project_root)
        splits = ("train", "test") if split == "both" else (split,)
        self.records = discover_modelnet40_records(root, splits=splits)
        if categories is not None:
            cat_set = set(categories)
            self.records = [r for r in self.records if r["category"] in cat_set]
        self._build_category_table()

    def _build_category_table(self) -> None:
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


class RenderedModelNetDataset(RenderedMeshDataset):
    """ModelNet40 + multi-view RGB / depth / normal (see ``RenderedMeshDataset``)."""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        split: Literal["train", "test", "both"] = "train",
        categories: Optional[Sequence[str]] = None,
        render_cfg: Optional[RenderConfig] = None,
        flatten_views: bool = True,
        project_root: Optional[Union[str, Path]] = None,
        max_cache_objects: int = 64,
    ) -> None:
        root = Path(root) if root is not None else default_modelnet40_root(project_root)
        splits = ("train", "test") if split == "both" else (split,)
        records = discover_modelnet40_records(root, splits=splits)
        if categories is not None:
            cat_set = set(categories)
            records = [r for r in records if r["category"] in cat_set]
        super().__init__(
            records,
            render_cfg=render_cfg,
            flatten_views=flatten_views,
            max_cache_objects=max_cache_objects,
        )

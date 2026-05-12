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

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from src.datasets.modelnet40_index import (
    default_modelnet40_root,
    discover_modelnet40_records,
    load_records_json,
    save_records_json,
)
from src.datasets.rendered_mesh_dataset import RenderedMeshDataset
from src.rendering.mesh_renderer import RenderConfig


__all__ = [
    "ModelNet40MeshDataset",
    "RenderedModelNetDataset",
    "default_modelnet40_root",
    "discover_modelnet40_records",
    "load_records_json",
    "save_records_json",
]


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

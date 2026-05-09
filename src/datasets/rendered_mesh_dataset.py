"""
Wrap mesh **records** (ModelNet, synthetic, future ShapeNet) with multi-view rendering.

Each **record** must contain: mesh_path, category, split, object_id, dataset (optional).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.rendering.mesh_renderer import MeshMultiviewRenderer, RenderConfig


class RenderedMeshDataset(Dataset):
    """
    If ``flatten_views`` is True (default), ``len = len(records) * n_views`` and each
    item is one view (scalar ``view_id``), matching the project sample schema.

    Renders are **cached per object_id** in memory to avoid re-rendering every view access.
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        render_cfg: Optional[RenderConfig] = None,
        flatten_views: bool = True,
        max_cache_objects: int = 64,
    ) -> None:
        self.records = list(records)
        self.render_cfg = render_cfg or RenderConfig()
        self.flatten_views = flatten_views
        self.n_views = int(self.render_cfg.n_views)
        self._renderer = MeshMultiviewRenderer(self.render_cfg)
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_order: List[str] = []
        self._max_cache = int(max_cache_objects)

    def __len__(self) -> int:
        if self.flatten_views:
            return len(self.records) * self.n_views
        return len(self.records)

    def _cache_put(self, oid: str, views: List[Dict[str, Any]]) -> None:
        if oid in self._cache:
            return
        while len(self._cache_order) >= self._max_cache:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        self._cache[oid] = views
        self._cache_order.append(oid)

    def _render_record(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        oid = str(rec.get("object_id", rec["mesh_path"]))
        if oid in self._cache:
            return self._cache[oid]
        mesh_arg: Union[str, Any] = rec.get("mesh_path") or ""
        if rec.get("mesh") is not None:
            mesh_arg = rec["mesh"]
        elif not mesh_arg:
            raise ValueError("Record needs mesh_path or in-memory 'mesh' (trimesh.Trimesh).")
        views = self._renderer.render_mesh(mesh_arg, mesh_path_for_meta=str(rec.get("mesh_path", "")))
        self._cache_put(oid, views)
        return views

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.flatten_views:
            rec_i = index // self.n_views
            view_id = index % self.n_views
        else:
            rec_i = index
            view_id = -1

        rec = self.records[rec_i]
        views = self._render_record(rec)

        if self.flatten_views:
            v = views[view_id]
            return self._pack_sample(rec, view_id, v)

        rgb = np.stack([x["rgb"] for x in views], axis=0)
        depth = np.stack([x["depth"] for x in views], axis=0)
        normal = np.stack([x["normal"] for x in views], axis=0)
        view_ids = np.arange(self.n_views, dtype=np.int64)
        return {
            **self._meta(rec),
            "view_id": view_ids,
            "rgb": rgb,
            "depth": depth,
            "normal": normal,
        }

    @staticmethod
    def _meta(rec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mesh_path": rec.get("mesh_path", ""),
            "category": rec["category"],
            "split": rec["split"],
            "object_id": rec.get("object_id", ""),
            "dataset": rec.get("dataset", "mesh"),
        }

    def _pack_sample(self, rec: Dict[str, Any], view_id: int, v: Dict[str, Any]) -> Dict[str, Any]:
        out = self._meta(rec)
        out["view_id"] = int(view_id)
        out["rgb"] = v["rgb"]
        out["depth"] = v["depth"]
        out["normal"] = v["normal"]
        return out


def collate_rendered_batch(
    batch: List[Dict[str, Any]],
    *,
    to_torch: bool = True,
) -> Dict[str, Any]:
    """Optional collate: stacks rgb/depth/normal to tensors."""
    keys = {"mesh_path", "category", "split", "view_id", "object_id", "dataset"}
    out: Dict[str, Any] = {k: [b[k] for b in batch] for k in keys if k in batch[0]}
    rgb = np.stack([b["rgb"] for b in batch], axis=0)
    depth = np.stack([b["depth"] for b in batch], axis=0)
    normal = np.stack([b["normal"] for b in batch], axis=0)
    if to_torch:
        out["rgb"] = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
        out["depth"] = torch.from_numpy(depth).unsqueeze(1)
        out["normal"] = torch.from_numpy(normal).permute(0, 3, 1, 2)
    else:
        out["rgb"] = rgb
        out["depth"] = depth
        out["normal"] = normal
    return out

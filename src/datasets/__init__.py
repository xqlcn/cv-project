"""Mesh datasets: ModelNet40, synthetic primitives, rendered wrappers."""

from src.datasets.modelnet40_dataset import (
    ModelNet40MeshDataset,
    RenderedModelNetDataset,
    default_modelnet40_root,
    discover_modelnet40_records,
)
from src.datasets.rendered_mesh_dataset import RenderedMeshDataset, collate_rendered_batch
from src.datasets.synthetic_primitives import (
    RenderedSyntheticPrimitiveDataset,
    SyntheticPrimitiveMeshDataset,
    build_synthetic_primitive_records,
    default_synthetic_root,
)

__all__ = [
    "ModelNet40MeshDataset",
    "RenderedModelNetDataset",
    "RenderedSyntheticPrimitiveDataset",
    "SyntheticPrimitiveMeshDataset",
    "RenderedMeshDataset",
    "collate_rendered_batch",
    "default_modelnet40_root",
    "default_synthetic_root",
    "discover_modelnet40_records",
    "build_synthetic_primitive_records",
]

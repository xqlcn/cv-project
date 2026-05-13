"""Mesh datasets: synthetic primitives and rendered wrappers."""

from importlib import import_module

__all__ = [
    "RenderedSyntheticPrimitiveDataset",
    "SyntheticPrimitiveMeshDataset",
    "RenderedMeshDataset",
    "collate_rendered_batch",
    "default_synthetic_root",
    "build_synthetic_primitive_records",
]


def __getattr__(name: str):
    if name in {"RenderedMeshDataset", "collate_rendered_batch"}:
        rendered_mesh_dataset = import_module("src.datasets.rendered_mesh_dataset")
        return getattr(rendered_mesh_dataset, name)
    if name in {
        "RenderedSyntheticPrimitiveDataset",
        "SyntheticPrimitiveMeshDataset",
        "build_synthetic_primitive_records",
        "default_synthetic_root",
    }:
        synthetic_primitives = import_module("src.datasets.synthetic_primitives")
        return getattr(synthetic_primitives, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

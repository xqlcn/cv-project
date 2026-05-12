"""Mesh datasets: ModelNet40, synthetic primitives, rendered wrappers."""

from importlib import import_module

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
    "save_records_json",
    "load_records_json",
    "build_synthetic_primitive_records",
]


def __getattr__(name: str):
    if name in {
        "default_modelnet40_root",
        "discover_modelnet40_records",
        "save_records_json",
        "load_records_json",
    }:
        modelnet40_index = import_module("src.datasets.modelnet40_index")
        return getattr(modelnet40_index, name)
    if name in {"ModelNet40MeshDataset", "RenderedModelNetDataset"}:
        modelnet40_dataset = import_module("src.datasets.modelnet40_dataset")
        return getattr(modelnet40_dataset, name)
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

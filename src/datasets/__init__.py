"""Mesh datasets: synthetic primitives, ModelNet40, and rendered wrappers."""

from importlib import import_module

__all__ = [
    "RenderedSyntheticPrimitiveDataset",
    "SyntheticPrimitiveMeshDataset",
    "RenderedModelNetDataset",
    "ModelNet40MeshDataset",
    "RenderedMeshDataset",
    "collate_rendered_batch",
    "default_synthetic_root",
    "default_modelnet40_root",
    "discover_modelnet40_records",
    "save_records_json",
    "load_records_json",
    "build_synthetic_primitive_records",
]

_MODELNET_NAMES = {
    "ModelNet40MeshDataset",
    "RenderedModelNetDataset",
    "default_modelnet40_root",
    "discover_modelnet40_records",
    "load_records_json",
    "save_records_json",
}

_RENDERED_MESH_NAMES = {"RenderedMeshDataset", "collate_rendered_batch"}

_SYNTHETIC_NAMES = {
    "RenderedSyntheticPrimitiveDataset",
    "SyntheticPrimitiveMeshDataset",
    "build_synthetic_primitive_records",
    "default_synthetic_root",
}


def __getattr__(name: str):
    if name in _MODELNET_NAMES:
        modelnet40_dataset = import_module("src.datasets.modelnet40_dataset")
        return getattr(modelnet40_dataset, name)
    if name in _RENDERED_MESH_NAMES:
        rendered_mesh_dataset = import_module("src.datasets.rendered_mesh_dataset")
        return getattr(rendered_mesh_dataset, name)
    if name in _SYNTHETIC_NAMES:
        synthetic_primitives = import_module("src.datasets.synthetic_primitives")
        return getattr(synthetic_primitives, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

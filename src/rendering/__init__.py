"""Multi-view mesh rendering (RGB, depth, normals) for dataset pipelines."""

from src.rendering.mesh_renderer import MeshMultiviewRenderer, render_mesh_multiview

__all__ = ["MeshMultiviewRenderer", "render_mesh_multiview"]

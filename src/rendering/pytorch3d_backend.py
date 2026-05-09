"""
Optional **PyTorch3D** backend (same role as ``mesh_renderer.py``).

PyTorch3D wheels are platform/CUDA-specific and are not pinned in ``requirements.txt``.
Implement a class mirroring :class:`src.rendering.mesh_renderer.MeshMultiviewRenderer`
here if you install PyTorch3D for your environment.

Until then, use ``MeshMultiviewRenderer`` (trimesh + pyrender) from ``mesh_renderer.py``.
"""

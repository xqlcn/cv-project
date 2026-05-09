"""
Multi-view mesh rendering: RGB (lit), depth (meters), normals (camera space, [-1,1]).

Uses **trimesh** + **pyrender** (CPU/GPU-friendly; works without PyTorch3D).
Optional PyTorch3D backend can be added later behind the same API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from src.rendering.camera_utils import fibonacci_sphere_views, orbit_views


@dataclass
class RenderConfig:
    image_size: Tuple[int, int] = (224, 224)
    n_views: int = 8
    camera_mode: Literal["fibonacci", "orbit"] = "fibonacci"
    distance: float = 2.2
    light_intensity: float = 3.0
    z_near: float = 0.05
    z_far: float = 10.0
    yfov_deg: float = 50.0
    seed: int = 0


def _load_trimesh(path: str):
    import trimesh

    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError(f"No mesh geometry in scene: {path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(loaded)} for {path}")
    return loaded


def _center_scale_mesh(mesh: "trimesh.Trimesh", target_extent: float = 1.0) -> "trimesh.Trimesh":
    import trimesh

    mesh = mesh.copy()
    mesh.vertices -= mesh.vertices.mean(axis=0)
    ext = np.ptp(mesh.vertices, axis=0).max()
    if ext > 1e-8:
        mesh.vertices *= target_extent / ext
    return mesh


def _world_to_camera_rot(camera_to_world: np.ndarray) -> np.ndarray:
    """Rotation R_wc such that v_cam = R_wc @ v_world for directions (orthogonal R)."""
    R_cw = camera_to_world[:3, :3]
    return R_cw.T


def _encode_normal_vertex_colors(mesh: "trimesh.Trimesh", camera_to_world: np.ndarray) -> "trimesh.Trimesh":
    """Copy mesh with vertex colors = camera-space normals mapped to [0,255] (unlit visualization)."""
    import trimesh

    m = mesh.copy()
    vn = m.vertex_normals.astype(np.float64)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    R_wc = _world_to_camera_rot(camera_to_world)
    n_cam = (R_wc @ vn.T).T
    rgb = ((n_cam * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    m.visual = trimesh.visual.ColorVisuals(vertex_colors=rgb)
    return m


class MeshMultiviewRenderer:
    """
    Renders a mesh from multiple viewpoints.

    Outputs per view:
      - rgb: uint8 [H,W,3] lit shading
      - depth: float32 [H,W] linear depth in meters (inf = background)
      - normal: float32 [H,W,3] camera-space unit normals in [-1,1] (from vertex interp)
    """

    def __init__(self, cfg: Optional[RenderConfig] = None) -> None:
        self.cfg = cfg or RenderConfig()
        self._renderer = None
        self._size: Optional[Tuple[int, int]] = None

    def _get_renderer(self, width: int, height: int):
        import pyrender

        if self._renderer is None or self._size != (width, height):
            self._renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            self._size = (width, height)
        return self._renderer

    def _poses(self) -> List[np.ndarray]:
        h, w = self.cfg.image_size
        n = self.cfg.n_views
        if self.cfg.camera_mode == "orbit":
            return orbit_views(n, radius=self.cfg.distance)
        return fibonacci_sphere_views(
            n,
            radius=self.cfg.distance,
            seed=self.cfg.seed,
        )

    def render_mesh(
        self,
        mesh_input: Union[str, Any],
        *,
        mesh_path_for_meta: str = "",
    ) -> List[Dict[str, Any]]:
        """
        mesh_input: path to .off/.obj/etc. or a **trimesh.Trimesh** (centered internally).
        """
        import pyrender
        import trimesh

        if isinstance(mesh_input, str):
            mesh = _load_trimesh(mesh_input)
            mpath = mesh_input
        else:
            mesh = mesh_input
            mpath = mesh_path_for_meta or ""

        mesh = _center_scale_mesh(mesh)
        width, height = self.cfg.image_size
        renderer = self._get_renderer(width, height)
        yfov = np.deg2rad(self.cfg.yfov_deg)
        cam = pyrender.PerspectiveCamera(
            yfov=yfov,
            aspectRatio=float(width) / float(height),
            znear=float(self.cfg.z_near),
            zfar=float(self.cfg.z_far),
        )

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.cfg.light_intensity)

        poses = self._poses()
        out_views: List[Dict[str, Any]] = []

        for view_id, pose in enumerate(poses):
            # --- Lit RGB mesh ---
            pr_mesh_rgb = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            scene = pyrender.Scene(ambient_light=[0.15, 0.15, 0.18], bg_color=[0.0, 0.0, 0.0, 0.0])
            scene.add(pr_mesh_rgb)
            scene.add(cam, pose=pose)
            scene.add(light, pose=pose)
            color, depth = renderer.render(scene)
            rgb = np.asarray(color[:, :, :3], dtype=np.uint8)
            depth_m = np.asarray(depth, dtype=np.float32)

            # --- Normal visualization mesh (vertex colors = camera-space normals) ---
            mesh_n = _encode_normal_vertex_colors(mesh, pose)
            pr_mesh_n = pyrender.Mesh.from_trimesh(mesh_n, smooth=True)
            # Ambient-only so vertex colors (camera-space normals) are not washed by directional BRDF.
            scene_n = pyrender.Scene(ambient_light=[10.0, 10.0, 10.0], bg_color=[0.5, 0.5, 0.5, 1.0])
            scene_n.add(pr_mesh_n)
            scene_n.add(cam, pose=pose)
            color_n, _ = renderer.render(scene_n)
            normal_rgb = np.asarray(color_n[:, :, :3], dtype=np.float32) / 255.0
            normal = normal_rgb * 2.0 - 1.0
            norms = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
            normal = normal / np.clip(norms, 1e-8, None)

            out_views.append(
                {
                    "view_id": int(view_id),
                    "rgb": rgb,
                    "depth": depth_m,
                    "normal": normal.astype(np.float32),
                    "camera_pose": pose.astype(np.float32),
                }
            )

        return out_views


def render_mesh_multiview(
    mesh_input: Union[str, Any],
    *,
    cfg: Optional[RenderConfig] = None,
    mesh_path: str = "",
) -> List[Dict[str, Any]]:
    """Functional wrapper."""
    r = MeshMultiviewRenderer(cfg)
    return r.render_mesh(mesh_input, mesh_path_for_meta=mesh_path)

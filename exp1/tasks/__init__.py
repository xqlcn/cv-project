"""Label builders for Experiment 1 probing tasks."""

from __future__ import annotations

from exp1.tasks.camera import build_camera_labels
from exp1.tasks.dense_surface_normals import Exp1DenseSurfaceNormalDataset
from exp1.tasks.lighting import build_lighting_labels
from exp1.tasks.relative_depth import build_relative_depth_labels
from exp1.tasks.scale import build_scale_labels

__all__ = [
    "Exp1DenseSurfaceNormalDataset",
    "build_camera_labels",
    "build_lighting_labels",
    "build_relative_depth_labels",
    "build_scale_labels",
]

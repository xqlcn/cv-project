"""Camera poses on a sphere (world frame); compatible with pyrender / future PyTorch3D."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def look_at_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64),
) -> np.ndarray:
    """
    Build 4x4 **camera-to-world** transform (pyrender convention: maps camera coords → world).

    Camera looks along **-Z** in camera space, +Y up, +X right (OpenGL-style).
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)

    # Columns: right, up, -forward (camera -Z is forward direction in world)
    R = np.stack([right, true_up, -forward], axis=1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def fibonacci_sphere_views(
    n_views: int,
    *,
    radius: float = 2.2,
    elevation_jitter: float = 0.08,
    seed: int = 0,
) -> List[np.ndarray]:
    """Return n_views camera-to-world 4x4 matrices on a sphere around origin."""
    rng = np.random.default_rng(seed)
    poses: List[np.ndarray] = []
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n_views):
        t = float(i) / max(n_views, 1)
        z = 1.0 - 2.0 * t
        z = max(-1.0, min(1.0, z))
        r_xy = math.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * math.pi * golden * float(i)
        x = r_xy * math.cos(theta)
        y = r_xy * math.sin(theta)
        el = elevation_jitter * rng.standard_normal()
        p = np.array([x, y, z], dtype=np.float64) * radius
        p[1] += el * radius * 0.1
        pose = look_at_rotation(p, np.zeros(3, dtype=np.float64))
        poses.append(pose)
    return poses


def orbit_views(
    n_views: int,
    *,
    radius: float = 2.5,
    elevation: float = 0.25,
    start_azimuth: float = 0.0,
) -> List[np.ndarray]:
    """Evenly spaced orbit at fixed elevation (radians)."""
    poses: List[np.ndarray] = []
    for i in range(n_views):
        az = start_azimuth + (2.0 * math.pi * i) / max(n_views, 1)
        x = radius * math.cos(elevation) * math.cos(az)
        y = radius * math.sin(elevation)
        z = radius * math.cos(elevation) * math.sin(az)
        eye = np.array([x, y, z], dtype=np.float64)
        poses.append(look_at_rotation(eye, np.zeros(3, dtype=np.float64)))
    return poses

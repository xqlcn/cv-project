from __future__ import annotations

import numpy as np

from exp1.data.label_join import join_labels_by_render_id
from exp1.tasks.camera import build_camera_labels
from exp1.tasks.lighting import build_lighting_labels
from exp1.tasks.relative_depth import (
    build_relative_depth_labels,
    relative_depth_label_row,
)
from exp1.tasks.scale import build_scale_labels
from exp1.tasks.surface_normals import (
    aggregate_surface_normal_label,
    build_surface_normal_aggregate_labels,
)


def test_aggregate_surface_normal_label_normalizes_foreground_mean() -> None:
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    mask = np.ones((4, 4), dtype=bool)

    label = aggregate_surface_normal_label(normal, mask, min_valid_pixels=1)

    assert label["label_valid"] is True
    assert label["mean_normal_x"] == 0.0
    assert label["mean_normal_y"] == 0.0
    assert label["mean_normal_z"] == 1.0


def test_relative_depth_label_row_uses_region_pairs_and_valid_masks() -> None:
    depth = np.zeros((6, 6), dtype=np.float32)
    for region in range(9):
        row = region // 3
        col = region % 3
        depth[row * 2 : row * 2 + 2, col * 2 : col * 2 + 2] = float(region + 1)
    mask = np.ones((6, 6), dtype=bool)

    label = relative_depth_label_row(
        depth,
        mask,
        region_pairs=[(0, 1), (1, 0)],
        grid_size=(3, 3),
        min_depth_margin=0.1,
    )

    assert label["pair_0_label"] == 1
    assert label["pair_0_valid"] is True
    assert label["pair_1_label"] == 0
    assert label["relative_depth_valid_pair_count"] == 2


def test_build_labels_from_buffers_and_join_by_render_id(tmp_path) -> None:
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 1] = 1.0
    mask = np.ones((4, 4), dtype=bool)
    depth = np.ones((4, 4), dtype=np.float32)
    normal_path = tmp_path / "normal.npy"
    mask_path = tmp_path / "mask.npy"
    depth_path = tmp_path / "depth.npy"
    np.save(normal_path, normal)
    np.save(mask_path, mask)
    np.save(depth_path, depth)
    rows = [
        {
            "render_id": "r1",
            "object_id": "obj",
            "normal_path": str(normal_path),
            "depth_path": str(depth_path),
            "mask_path": str(mask_path),
        }
    ]

    snorm = build_surface_normal_aggregate_labels(rows, min_valid_pixels=1)
    rdepth = build_relative_depth_labels(
        rows,
        region_pairs=[(0, 1)],
        grid_size=(2, 2),
        min_depth_margin=0.1,
    )
    joined = join_labels_by_render_id(snorm, rdepth)

    assert joined.loc[0, "render_id"] == "r1"
    assert joined.loc[0, "mean_normal_y"] == 1.0
    assert joined.loc[0, "relative_depth_valid_pair_count"] == 0


def test_metadata_label_builders() -> None:
    rows = [
        {
            "render_id": "r1",
            "camera_distance": 2.0,
            "camera_azimuth_deg": 90.0,
            "camera_elevation_deg": 0.0,
            "camera_fov_deg": 50.0,
            "light_type": "sun",
            "light_azimuth_deg": 0.0,
            "light_elevation_deg": 90.0,
            "light_intensity": 4.0,
            "object_scale": 1.25,
            "qc_foreground_fraction": 0.2,
        }
    ]

    camera = build_camera_labels(rows)
    lighting = build_lighting_labels(rows)
    scale = build_scale_labels(rows)

    assert camera.loc[0, "azimuth_sin"] == 1.0
    assert lighting.loc[0, "light_dir_z"] == 1.0
    assert scale.loc[0, "foreground_area_fraction"] == 0.2

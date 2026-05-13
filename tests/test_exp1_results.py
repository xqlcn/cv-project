from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from exp1.analysis.plots import plot_layerwise_metrics, plot_texture_drops
from exp1.analysis.qualitative import make_qualitative_probe_table
from exp1.evaluation.comparisons import (
    compute_texture_dependence_drops,
    metric_direction,
)
from exp1.evaluation.metrics import (
    aggregate_prediction_bootstrap_cis,
    aggregate_probe_metrics,
    bootstrap_mean_ci,
)


def _write_metrics(path, *, task, model, layer, texture, metric_name, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "task": task,
            "model_name": model,
            "layer_name": layer,
            "texture_condition": texture,
        },
        "metrics": {"test": {metric_name: value}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_probe_metrics_builds_long_table(tmp_path) -> None:
    probe_root = tmp_path / "probes"
    _write_metrics(
        probe_root
        / "clip_vit_b16"
        / "final"
        / "surface_normal_aggregate"
        / "metrics.json",
        task="surface_normal_aggregate",
        model="clip_vit_b16",
        layer="final",
        texture="photorealistic",
        metric_name="angular_error_deg_mean",
        value=20.0,
    )

    table = aggregate_probe_metrics(probe_root)

    assert list(table["task"]) == ["surface_normal_aggregate"]
    assert list(table["model"]) == ["clip_vit_b16"]
    assert list(table["split"]) == ["test"]
    assert list(table["metric"]) == ["angular_error_deg_mean"]
    assert list(table["value"]) == [20.0]


def test_aggregate_probe_metrics_labels_cross_texture_runs(tmp_path) -> None:
    probe_root = tmp_path / "probes"
    path = (
        probe_root
        / "clip_vit_b16"
        / "final"
        / "surface_normal_aggregate"
        / "train_flat__test_random_noise"
        / "metrics.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "task": "surface_normal_aggregate",
                    "model_name": "clip_vit_b16",
                    "layer_name": "final",
                    "train_texture_condition": ["flat"],
                    "eval_texture_condition": ["random_noise"],
                },
                "metrics": {"test": {"angular_error_deg_mean": 30.0}},
            }
        ),
        encoding="utf-8",
    )

    table = aggregate_probe_metrics(probe_root)

    assert list(table["texture_condition"]) == ["train_flat__test_random_noise"]
    assert list(table["train_texture_condition"]) == ["flat"]
    assert list(table["eval_texture_condition"]) == ["random_noise"]


def test_texture_dependence_drops_respect_metric_direction() -> None:
    results = pd.DataFrame(
        [
            {
                "task": "surface_normal_aggregate",
                "model": "clip",
                "layer": "final",
                "texture_condition": "photorealistic",
                "split": "test",
                "metric": "angular_error_deg_mean",
                "value": 20.0,
            },
            {
                "task": "surface_normal_aggregate",
                "model": "clip",
                "layer": "final",
                "texture_condition": "flat",
                "split": "test",
                "metric": "angular_error_deg_mean",
                "value": 25.0,
            },
            {
                "task": "relative_depth_regions",
                "model": "clip",
                "layer": "final",
                "texture_condition": "photorealistic",
                "split": "test",
                "metric": "valid_pair_accuracy",
                "value": 0.8,
            },
            {
                "task": "relative_depth_regions",
                "model": "clip",
                "layer": "final",
                "texture_condition": "random_noise",
                "split": "test",
                "metric": "valid_pair_accuracy",
                "value": 0.6,
            },
        ]
    )

    drops = compute_texture_dependence_drops(results)

    by_task = {row.task: row.texture_drop for row in drops.itertuples()}
    assert by_task["surface_normal_aggregate"] == 5.0
    assert by_task["relative_depth_regions"] == pytest.approx(0.2)


def test_metric_direction_covers_dense_depth_metrics() -> None:
    assert metric_direction("abs_rel_median") == "lower"
    assert metric_direction("ssi_l1_median") == "lower"
    assert metric_direction("pearson_r_mean") == "higher"
    assert metric_direction("delta_1_mean") == "higher"


def test_texture_dependence_drops_ignore_cross_texture_by_default() -> None:
    results = pd.DataFrame(
        [
            {
                "task": "relative_depth_regions",
                "model": "clip",
                "layer": "final",
                "texture_condition": "photorealistic",
                "train_texture_condition": "all",
                "eval_texture_condition": "all",
                "split": "test",
                "metric": "valid_pair_accuracy",
                "value": 0.8,
            },
            {
                "task": "relative_depth_regions",
                "model": "clip",
                "layer": "final",
                "texture_condition": "flat",
                "train_texture_condition": "all",
                "eval_texture_condition": "all",
                "split": "test",
                "metric": "valid_pair_accuracy",
                "value": 0.7,
            },
            {
                "task": "relative_depth_regions",
                "model": "clip",
                "layer": "final",
                "texture_condition": "train_flat__test_random_noise",
                "train_texture_condition": "flat",
                "eval_texture_condition": "random_noise",
                "split": "test",
                "metric": "valid_pair_accuracy",
                "value": 0.4,
            },
        ]
    )

    drops = compute_texture_dependence_drops(results)

    assert drops["comparison_texture"].tolist() == ["flat"]


def test_texture_dependence_drops_can_include_cross_texture() -> None:
    results = pd.DataFrame(
        [
            {
                "task": "relative_depth_regions",
                "model": "clip",
                "layer": "final",
                "texture_condition": "photorealistic",
                "train_texture_condition": "all",
                "eval_texture_condition": "all",
                "split": "test",
                "metric": "valid_pair_accuracy",
                "value": 0.8,
            },
            {
                "task": "relative_depth_regions",
                "model": "clip",
                "layer": "final",
                "texture_condition": "train_flat__test_random_noise",
                "train_texture_condition": "flat",
                "eval_texture_condition": "random_noise",
                "split": "test",
                "metric": "valid_pair_accuracy",
                "value": 0.4,
            },
        ]
    )

    drops = compute_texture_dependence_drops(results, include_cross_texture=True)

    assert drops["comparison_texture"].tolist() == [
        "train_flat__test_random_noise"
    ]


def test_layerwise_plot_is_generated(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "task": "surface_normal_aggregate",
                "model": "clip",
                "layer": layer,
                "texture_condition": "flat",
                "split": "test",
                "metric": "angular_error_deg_mean",
                "value": value,
            }
            for layer, value in [("layer4", 25.0), ("layer8", 22.0), ("final", 24.0)]
        ]
    )

    paths = plot_layerwise_metrics(results, tmp_path)

    assert len(paths) == 1
    assert paths[0].is_file()


def test_texture_drop_plot_defaults_to_test_split(tmp_path) -> None:
    drops = pd.DataFrame(
        [
            {
                "task": "relative_depth_regions",
                "model": "clip",
                "layer": "final",
                "split": split,
                "metric": "valid_pair_accuracy",
                "comparison_texture": "flat",
                "texture_drop": value,
            }
            for split, value in [("train", 0.1), ("test", 0.2)]
        ]
    )

    paths = plot_texture_drops(drops, tmp_path)

    assert len(paths) == 1
    assert "test" in paths[0].name


def test_qualitative_probe_table_is_generated_for_surface_normals(tmp_path) -> None:
    textures = ["photorealistic", "flat", "random_noise"]
    rows = []
    label_rows = []
    probe_root = tmp_path / "probes"
    for idx, texture in enumerate(textures):
        render_id = f"render_{idx}"
        rgb_path = tmp_path / f"{texture}.png"
        mask_path = tmp_path / f"{texture}_mask.npy"
        image = Image.new("RGB", (16, 16), (40 + idx * 40, 80, 120))
        image.save(rgb_path)
        np.save(mask_path, np.ones((16, 16), dtype=bool))
        rows.append(
            {
                "render_id": render_id,
                "split": "test",
                "texture_condition": texture,
                "texture_control_group_id": "group_0",
                "rgb_path": str(rgb_path),
                "mask_path": str(mask_path),
            }
        )
        label_rows.append(
            {
                "render_id": render_id,
                "mean_normal_x": 0.0,
                "mean_normal_y": 0.0,
                "mean_normal_z": 1.0,
                "label_valid": True,
            }
        )
        pred_dir = (
            probe_root
            / "clip_vit_b16"
            / "final"
            / "surface_normal_aggregate"
            / f"texture_{texture}"
        )
        pred_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "render_id": render_id,
                    "split": "test",
                    "pred_mean_normal_x": 0.0,
                    "pred_mean_normal_y": 0.0,
                    "pred_mean_normal_z": 1.0,
                }
            ]
        ).to_csv(pred_dir / "predictions.csv", index=False)

    manifest_path = tmp_path / "manifest.csv"
    label_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    pd.DataFrame(label_rows).to_csv(label_path, index=False)

    output = make_qualitative_probe_table(
        task="surface_normal_aggregate",
        manifest_path=manifest_path,
        label_path=label_path,
        probe_root=probe_root,
        output_path=tmp_path / "qualitative.png",
        project_root=tmp_path,
        model_display_order={"clip_vit_b16": "CLIP B/16"},
        textures=textures,
    )

    assert output.is_file()


def test_qualitative_probe_table_is_generated_for_relative_depth(tmp_path) -> None:
    textures = ["photorealistic", "flat", "random_noise"]
    rows = []
    label_rows = []
    probe_root = tmp_path / "probes"
    for idx, texture in enumerate(textures):
        render_id = f"render_{idx}"
        rgb_path = tmp_path / f"{texture}.png"
        Image.new("RGB", (16, 16), (40, 80 + idx * 40, 120)).save(rgb_path)
        rows.append(
            {
                "render_id": render_id,
                "split": "test",
                "texture_condition": texture,
                "texture_control_group_id": "group_0",
                "rgb_path": str(rgb_path),
            }
        )
        label_rows.append(
            {
                "render_id": render_id,
                "relative_depth_grid_rows": 3,
                "relative_depth_grid_cols": 3,
                "relative_depth_pair_count": 1,
                "pair_0_region_a": 0,
                "pair_0_region_b": 1,
                "pair_0_label": 1,
                "pair_0_valid": True,
                "label_valid": True,
            }
        )
        pred_dir = (
            probe_root
            / "clip_vit_b16"
            / "final"
            / "relative_depth_regions"
            / f"texture_{texture}"
        )
        pred_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "render_id": render_id,
                    "split": "test",
                    "pair_0_prob": 0.75,
                }
            ]
        ).to_csv(pred_dir / "predictions.csv", index=False)

    manifest_path = tmp_path / "manifest.csv"
    label_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    pd.DataFrame(label_rows).to_csv(label_path, index=False)

    output = make_qualitative_probe_table(
        task="relative_depth_regions",
        manifest_path=manifest_path,
        label_path=label_path,
        probe_root=probe_root,
        output_path=tmp_path / "qualitative_depth.png",
        project_root=tmp_path,
        model_display_order={"clip_vit_b16": "CLIP B/16"},
        textures=textures,
    )

    assert output.is_file()


def test_bootstrap_mean_ci_can_resample_by_object() -> None:
    ci = bootstrap_mean_ci(
        [1.0, 2.0, 3.0, 4.0],
        units=["a", "a", "b", "b"],
        n_resamples=50,
        seed=1,
    )

    assert ci["mean"] == 2.5
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_aggregate_prediction_bootstrap_cis_uses_object_units(tmp_path) -> None:
    probe_dir = (
        tmp_path
        / "probes"
        / "clip"
        / "final"
        / "surface_normal_aggregate"
        / "texture_flat"
    )
    probe_dir.mkdir(parents=True)
    (probe_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "task": "surface_normal_aggregate",
                    "model_name": "clip",
                    "layer_name": "final",
                    "texture_condition": ["flat"],
                },
                "metrics": {"test": {"angular_error_deg_mean": 2.0}},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "render_id": ["r1", "r2", "r3"],
            "split": ["test", "test", "test"],
            "object_id": ["a", "a", "b"],
            "angular_error_deg": [1.0, 3.0, 5.0],
        }
    ).to_csv(probe_dir / "predictions.csv", index=False)

    ci = aggregate_prediction_bootstrap_cis(
        tmp_path / "probes",
        n_resamples=20,
        seed=3,
    )

    assert list(ci["metric"]) == ["angular_error_deg_mean"]
    assert ci.iloc[0]["mean"] == pytest.approx(3.0)
    assert ci.iloc[0]["unit_column"] == "object_id"

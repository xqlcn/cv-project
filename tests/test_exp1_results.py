from __future__ import annotations

import json

import pandas as pd
import pytest

from exp1.analysis.plots import plot_layerwise_metrics
from exp1.evaluation.comparisons import compute_texture_dependence_drops
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

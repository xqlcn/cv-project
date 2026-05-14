from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from exp1.features.storage import save_feature_cache
from exp1.probes.metrics import (
    regression_metrics,
    relative_depth_metrics,
    surface_normal_metrics,
    viewpoint_metrics,
)
from exp1.probes.train import ProbeTrainConfig, train_exp1_probe, train_probe_arrays


def test_surface_normal_metrics_report_angular_error() -> None:
    predictions = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    metrics = surface_normal_metrics(predictions, targets)

    assert metrics["angular_error_deg_mean"] == pytest.approx(45.0)
    assert metrics["angular_error_deg_median"] == pytest.approx(0.0)


def test_relative_depth_metrics_use_only_valid_pairs() -> None:
    logits = torch.tensor([[8.0, -8.0], [8.0, 8.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    valid = torch.tensor([[True, True], [False, True]])

    metrics = relative_depth_metrics(logits, targets, valid)

    assert metrics["valid_pair_count"] == 3.0
    assert metrics["valid_pair_accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_full_task_metrics_cover_regression_and_viewpoint() -> None:
    regression = regression_metrics(
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        torch.tensor([[1.5, 2.0], [2.5, 5.0]]),
    )
    viewpoint = viewpoint_metrics(
        torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
    )

    assert regression["mae_mean"] == pytest.approx(0.75)
    assert viewpoint["azimuth_angular_error_deg_mean"] == pytest.approx(0.0)
    assert viewpoint["elevation_angular_error_deg_mean"] == pytest.approx(90.0)


def test_lighting_direction_probe_overfits_synthetic_linear_labels() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(96, 3)).astype(np.float32)
    targets = features / np.linalg.norm(features, axis=1, keepdims=True)

    result = train_probe_arrays(
        features,
        targets,
        task="lighting_direction",
        config=ProbeTrainConfig(
            task="lighting_direction",
            epochs=120,
            batch_size=24,
            lr=5e-2,
            weight_decay=0.0,
            seed=3,
            device="cpu",
            use_layernorm=False,
            early_stopping=False,
        ),
    )

    assert result["metrics"]["train"]["angular_error_deg_mean"] < 3.0


def test_relative_depth_probe_overfits_synthetic_valid_pairs() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(128, 4)).astype(np.float32)
    targets = np.stack(
        [
            features[:, 0] + 0.25 * features[:, 1] > 0.0,
            features[:, 2] - features[:, 3] > 0.0,
        ],
        axis=1,
    ).astype(np.float32)
    valid = np.ones_like(targets, dtype=bool)

    result = train_probe_arrays(
        features,
        targets,
        task="relative_depth_regions",
        train_valid_mask=valid,
        config=ProbeTrainConfig(
            task="relative_depth_regions",
            epochs=100,
            batch_size=32,
            lr=5e-2,
            weight_decay=0.0,
            seed=5,
            device="cpu",
            use_layernorm=False,
            early_stopping=False,
        ),
    )

    assert result["metrics"]["train"]["valid_pair_accuracy"] > 0.95


def test_camera_distance_probe_overfits_synthetic_regression() -> None:
    rng = np.random.default_rng(13)
    features = rng.normal(size=(96, 3)).astype(np.float32)
    targets = (0.5 * features[:, :1] - 0.25 * features[:, 1:2]).astype(np.float32)

    result = train_probe_arrays(
        features,
        targets,
        task="camera_distance",
        config=ProbeTrainConfig(
            task="camera_distance",
            epochs=120,
            batch_size=24,
            lr=5e-2,
            weight_decay=0.0,
            seed=13,
            device="cpu",
            use_layernorm=False,
            early_stopping=False,
        ),
    )

    assert result["metrics"]["train"]["mae_mean"] < 0.05


def test_train_exp1_probe_checks_object_disjoint_splits_before_training(
    tmp_path,
) -> None:
    feature_path = tmp_path / "features.npz"
    label_path = tmp_path / "labels.csv"
    manifest_path = tmp_path / "manifest.csv"
    save_feature_cache(
        feature_path,
        render_ids=["r1", "r2"],
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    pd.DataFrame(
        [
            {
                "render_id": "r1",
                "log_camera_distance": 1.0,
                "label_valid": True,
            },
            {
                "render_id": "r2",
                "log_camera_distance": 2.0,
                "label_valid": True,
            },
        ]
    ).to_csv(label_path, index=False)
    pd.DataFrame(
        [
            {
                "render_id": "r1",
                "object_id": "leaky_object",
                "split": "train",
                "texture_condition": "flat",
            },
            {
                "render_id": "r2",
                "object_id": "leaky_object",
                "split": "test",
                "texture_condition": "flat",
            },
        ]
    ).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="multiple splits"):
        train_exp1_probe(
            feature_cache=feature_path,
            label_path=label_path,
            manifest_path=manifest_path,
            output_dir=tmp_path / "out",
            task="camera_distance",
            model_name="toy",
            layer_name="final",
            target_columns=["log_camera_distance"],
            config=ProbeTrainConfig(epochs=1, device="cpu"),
        )


def test_train_exp1_probe_saves_checkpoint_metrics_and_predictions(tmp_path) -> None:
    render_ids = [f"r{idx}" for idx in range(6)]
    features = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    targets = features / np.linalg.norm(features, axis=1, keepdims=True)
    feature_path = tmp_path / "features.npz"
    label_path = tmp_path / "labels.csv"
    manifest_path = tmp_path / "manifest.csv"
    save_feature_cache(feature_path, render_ids=render_ids, features=features)
    pd.DataFrame(
        [
            {
                "render_id": render_id,
                "light_dir_x": float(target[0]),
                "light_dir_y": float(target[1]),
                "light_dir_z": float(target[2]),
                "label_valid": True,
            }
            for render_id, target in zip(render_ids, targets)
        ]
    ).to_csv(label_path, index=False)
    pd.DataFrame(
        [
            {
                "render_id": render_id,
                "object_id": f"object_{idx}",
                "split": "train" if idx < 3 else "val" if idx < 5 else "test",
                "texture_condition": "flat",
            }
            for idx, render_id in enumerate(render_ids)
        ]
    ).to_csv(manifest_path, index=False)

    result = train_exp1_probe(
        feature_cache=feature_path,
        label_path=label_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        task="lighting_direction",
        model_name="toy",
        layer_name="final",
        target_columns=["light_dir_x", "light_dir_y", "light_dir_z"],
        config=ProbeTrainConfig(
            epochs=5,
            batch_size=2,
            lr=5e-2,
            weight_decay=0.0,
            device="cpu",
            use_layernorm=False,
            early_stopping=False,
        ),
    )

    assert set(result["metrics"]) == {"train", "val", "test"}
    for path in result["artifact_paths"].values():
        assert path.is_file()
    predictions = pd.read_csv(result["artifact_paths"]["predictions"])
    assert set(predictions["split"]) == {"train", "val", "test"}
    assert "pred_light_dir_z" in predictions.columns


def test_train_exp1_probe_supports_cross_texture_eval_filters(tmp_path) -> None:
    render_ids = [f"r{idx}" for idx in range(8)]
    features = np.eye(8, 3, dtype=np.float32)
    targets = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (8, 1))
    feature_path = tmp_path / "features.npz"
    label_path = tmp_path / "labels.csv"
    manifest_path = tmp_path / "manifest.csv"
    save_feature_cache(feature_path, render_ids=render_ids, features=features)
    pd.DataFrame(
        [
            {
                "render_id": render_id,
                "light_dir_x": 1.0,
                "light_dir_y": 0.0,
                "light_dir_z": 0.0,
                "label_valid": True,
            }
            for render_id in render_ids
        ]
    ).to_csv(label_path, index=False)
    pd.DataFrame(
        [
            {
                "render_id": render_id,
                "object_id": f"object_{idx}",
                "split": "train" if idx < 4 else "val" if idx < 6 else "test",
                "texture_condition": "flat" if idx < 6 else "random_noise",
            }
            for idx, render_id in enumerate(render_ids)
        ]
    ).to_csv(manifest_path, index=False)

    result = train_exp1_probe(
        feature_cache=feature_path,
        label_path=label_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        task="lighting_direction",
        model_name="toy",
        layer_name="final",
        target_columns=["light_dir_x", "light_dir_y", "light_dir_z"],
        train_texture_condition=["flat"],
        eval_texture_condition=["random_noise"],
        config=ProbeTrainConfig(
            epochs=1,
            batch_size=2,
            device="cpu",
            use_layernorm=False,
            early_stopping=False,
        ),
    )

    assert result["metadata"]["train_texture_condition"] == ["flat"]
    assert result["metadata"]["eval_texture_condition"] == ["random_noise"]
    assert result["metadata"]["num_rows_by_split"] == {
        "train": 4,
        "val": 2,
        "test": 2,
    }

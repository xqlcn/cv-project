from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from exp1.data.feature_dataset import Exp1FeatureDataset
from exp1.data.image_dataset import Exp1ImageDataset
from exp1.data.label_join import join_labels_by_render_id
from exp1.data.splits import SplitValidationError, assert_object_disjoint_splits


def test_assert_object_disjoint_splits_rejects_leakage() -> None:
    rows = pd.DataFrame(
        [
            {"render_id": "r1", "object_id": "obj", "split": "train"},
            {"render_id": "r2", "object_id": "obj", "split": "test"},
        ]
    )

    with pytest.raises(SplitValidationError, match="multiple splits"):
        assert_object_disjoint_splits(rows)


def test_image_dataset_filters_by_split_and_texture(tmp_path) -> None:
    image_path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
    rows = [
        {
            "render_id": "r_train_flat",
            "object_id": "obj1",
            "split": "train",
            "texture_condition": "flat",
            "rgb_path": str(image_path),
            "qc_pass": True,
        },
        {
            "render_id": "r_test_flat",
            "object_id": "obj2",
            "split": "test",
            "texture_condition": "flat",
            "rgb_path": str(image_path),
            "qc_pass": True,
        },
        {
            "render_id": "r_train_noise",
            "object_id": "obj3",
            "split": "train",
            "texture_condition": "random_noise",
            "rgb_path": str(image_path),
            "qc_pass": True,
        },
    ]

    dataset = Exp1ImageDataset(rows, split="train", texture_condition="flat")

    assert len(dataset) == 1
    assert dataset[0]["render_id"] == "r_train_flat"
    assert dataset[0]["image"].size == (4, 4)


def test_join_labels_by_render_id_ignores_row_order() -> None:
    manifest = pd.DataFrame(
        [
            {"render_id": "r1", "object_id": "obj1"},
            {"render_id": "r2", "object_id": "obj2"},
        ]
    )
    labels = pd.DataFrame(
        [
            {"render_id": "r2", "target": 20},
            {"render_id": "r1", "target": 10},
        ]
    )

    joined = join_labels_by_render_id(manifest, labels)

    assert list(joined["render_id"]) == ["r1", "r2"]
    assert list(joined["target"]) == [10, 20]


def test_relative_depth_feature_dataset_loads_target_and_valid_mask(tmp_path) -> None:
    feature_path = tmp_path / "features.npz"
    np.savez(
        feature_path,
        render_ids=np.asarray(["r1", "r2"]),
        features=np.asarray([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32),
    )
    labels = pd.DataFrame(
        [
            {
                "render_id": "r2",
                "pair_0_label": 0,
                "pair_0_valid": False,
                "pair_1_label": 1,
                "pair_1_valid": True,
                "label_valid": True,
            },
            {
                "render_id": "r1",
                "pair_0_label": 1,
                "pair_0_valid": True,
                "pair_1_label": 0,
                "pair_1_valid": True,
                "label_valid": True,
            },
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "render_id": "r2",
                "object_id": "obj2",
                "split": "test",
                "texture_condition": "flat",
            },
            {
                "render_id": "r1",
                "object_id": "obj1",
                "split": "train",
                "texture_condition": "flat",
            },
        ]
    )

    dataset = Exp1FeatureDataset(
        feature_path,
        labels,
        manifest=manifest,
        task="relative_depth_regions",
        split="test",
    )
    item = dataset[0]

    assert item["render_id"] == "r2"
    assert np.allclose(item["features"], [2.0, 2.5])
    assert np.array_equal(item["target"], np.asarray([0.0, 1.0], dtype=np.float32))
    assert np.array_equal(item["valid_mask"], np.asarray([False, True]))

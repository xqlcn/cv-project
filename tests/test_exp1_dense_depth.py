"""Unit tests for the dense per-patch relative-depth probe pipeline."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from exp1.features.storage import (
    load_patch_feature_cache,
    patch_feature_cache_path,
    save_patch_feature_cache,
)
from exp1.probes.dense_depth import (
    DenseDepthHead,
    DenseDepthHeadConfig,
    dense_depth_metrics,
    ssi_l1_loss,
)
from exp1.probes.train_dense_depth import (
    DenseDepthTrainConfig,
    train_dense_depth_probe,
)
from exp1.tasks.dense_depth import (
    Exp1DenseDepthDataset,
    align_depth_mask_to_model_input,
    nanaware_pool,
)


def test_nanaware_pool_handles_nans_and_mask() -> None:
    array = np.full((4, 4), np.nan, dtype=np.float32)
    array[2, 2] = 3.0
    array[3, 3] = 4.0
    mask = np.ones_like(array, dtype=bool)
    pooled, valid = nanaware_pool(array, grid_rows=2, grid_cols=2, mask=mask)
    assert pooled.shape == (2, 2)
    assert valid.shape == (2, 2)
    # Top-left and off-diagonal cells contain no finite values.
    assert not valid[0, 0]
    assert not valid[0, 1]
    assert not valid[1, 0]
    assert valid[1, 1]
    assert pooled[1, 1] == pytest.approx(3.5)

    # Mask should be respected: if we mask away the only finite cell, output is empty.
    mask2 = np.ones_like(array, dtype=bool)
    mask2[3, 3] = False
    pooled2, valid2 = nanaware_pool(
        array, grid_rows=2, grid_cols=2, mask=mask2
    )
    assert valid2[1, 1]
    assert pooled2[1, 1] == pytest.approx(3.0)


def test_ssi_l1_loss_is_scale_and_shift_invariant() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 4)
    mask = torch.ones(2, 4, 4, dtype=torch.bool)
    target = 3.0 * pred + 1.5
    loss_zero = ssi_l1_loss(pred, target, mask)
    loss_shift = ssi_l1_loss(pred + 10.0, target, mask)
    loss_scale = ssi_l1_loss(pred * 5.0, target, mask)
    assert float(loss_zero) == pytest.approx(0.0, abs=1e-5)
    assert float(loss_shift) == pytest.approx(0.0, abs=1e-5)
    assert float(loss_scale) == pytest.approx(0.0, abs=1e-5)


def test_dense_depth_metrics_perfect_after_alignment() -> None:
    torch.manual_seed(1)
    target = torch.rand(3, 4, 4) + 1.0  # strictly positive depths
    pred = target * 2.0 - 0.5  # arbitrary affine transform of the target
    mask = torch.ones(3, 4, 4, dtype=torch.bool)
    metrics = dense_depth_metrics(pred, target, mask)
    # Least-squares alignment recovers the affine, so AbsRel collapses to 0.
    assert metrics["scale_invariant_abs_rel_mean"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["scale_invariant_d1_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["pearson_r_mean"] == pytest.approx(1.0, abs=1e-5)


def test_dense_depth_metrics_handles_uncorrelated_predictions() -> None:
    rng = np.random.default_rng(0)
    target = torch.from_numpy(rng.uniform(1.0, 5.0, size=(4, 4, 4)).astype(np.float32))
    pred = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(4, 4, 4)).astype(np.float32))
    mask = torch.ones(4, 4, 4, dtype=torch.bool)
    metrics = dense_depth_metrics(pred, target, mask)
    # Random predictions give finite but worse-than-zero metrics.
    assert np.isfinite(metrics["abs_rel_mean"])
    assert 0.0 < metrics["delta_1_mean"] <= 1.0


def test_dense_depth_head_shapes() -> None:
    cfg = DenseDepthHeadConfig(feature_dim=8, use_layernorm=True)
    head = DenseDepthHead(cfg)
    features = torch.randn(2, 5, 5, 8)
    output = head(features)
    assert tuple(output.shape) == (2, 5, 5)


def _make_synthetic_render(
    base_dir: Path,
    render_id: str,
    split: str,
    texture: str,
    *,
    obj_id: str,
    image_size: int = 16,
) -> dict:
    folder = base_dir / "renders" / split / obj_id / render_id
    folder.mkdir(parents=True, exist_ok=True)
    seed = zlib.crc32(f"{render_id}|{split}|{texture}".encode("utf-8"))
    rng = np.random.default_rng(seed)
    depth = rng.uniform(2.0, 3.0, size=(image_size, image_size)).astype(np.float32)
    mask = np.zeros((image_size, image_size), dtype=bool)
    mask[2:14, 2:14] = True
    depth[~mask] = np.nan
    normal = np.zeros((image_size, image_size, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    np.save(folder / "depth.npy", depth)
    np.save(folder / "mask.npy", mask)
    np.save(folder / "normal_camera.npy", normal)
    return {
        "render_id": render_id,
        "split": split,
        "texture_condition": texture,
        "object_id": obj_id,
        "source_dataset": "synthetic",
        "category": "test",
        "depth_path": str(folder / "depth.npy"),
        "mask_path": str(folder / "mask.npy"),
        "normal_path": str(folder / "normal_camera.npy"),
        "rgb_path": str(folder / "rgb.png"),
    }


def _build_synthetic_manifest(base_dir: Path) -> pd.DataFrame:
    rows = []
    for split, obj_id in (("train", "objA"), ("train", "objB"), ("val", "objC"), ("test", "objD")):
        for texture in ("photorealistic", "flat", "random_noise"):
            render_id = f"r_{obj_id}_{texture}"
            rows.append(
                _make_synthetic_render(
                    base_dir,
                    render_id=render_id,
                    split=split,
                    texture=texture,
                    obj_id=obj_id,
                )
            )
    return pd.DataFrame(rows)


def test_dataset_pools_depth_and_yields_features(tmp_path: Path) -> None:
    manifest = _build_synthetic_manifest(tmp_path)
    cache_dir = tmp_path / "features"
    cache_path = patch_feature_cache_path(
        cache_dir, model_name="m", layer_name="final"
    )
    feature_dim = 6
    patch_side = 4
    render_ids = manifest["render_id"].tolist()
    patches = np.random.RandomState(0).randn(
        len(render_ids), patch_side, patch_side, feature_dim
    ).astype(np.float16)
    save_patch_feature_cache(
        cache_path,
        render_ids=render_ids,
        patch_features=patches,
        metadata={"layer_name": "final"},
    )

    dataset = Exp1DenseDepthDataset(
        cache_path,
        manifest=manifest,
        split="train",
        texture_condition=["photorealistic"],
    )
    assert len(dataset) == 2  # two train objects, photorealistic only
    sample = dataset[0]
    assert sample.features.shape == (patch_side, patch_side, feature_dim)
    assert sample.target.shape == (patch_side, patch_side)
    assert sample.valid.dtype == bool


def test_dataset_depth_targets_keep_image_orientation(tmp_path: Path) -> None:
    render = _make_synthetic_render(
        tmp_path,
        render_id="r_asymmetric",
        split="train",
        texture="photorealistic",
        obj_id="objA",
        image_size=8,
    )
    depth = np.arange(64, dtype=np.float32).reshape(8, 8)
    mask = np.ones((8, 8), dtype=bool)
    np.save(render["depth_path"], depth)
    np.save(render["mask_path"], mask)
    manifest = pd.DataFrame([render])

    patch_side = 4
    feature_dim = 2
    cache_path = patch_feature_cache_path(
        tmp_path / "features", model_name="m", layer_name="final"
    )
    save_patch_feature_cache(
        cache_path,
        render_ids=[render["render_id"]],
        patch_features=np.zeros((1, patch_side, patch_side, feature_dim), dtype=np.float32),
        metadata={"layer_name": "final"},
    )

    dataset = Exp1DenseDepthDataset(
        cache_path,
        manifest=manifest,
        split="train",
        texture_condition=["photorealistic"],
    )
    expected, expected_valid = nanaware_pool(
        depth,
        grid_rows=patch_side,
        grid_cols=patch_side,
        mask=mask,
    )
    sample = dataset[0]
    np.testing.assert_allclose(sample.target, expected)
    np.testing.assert_array_equal(sample.valid, expected_valid)


def _write_min_manifest_parquet(path: Path, manifest: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)
    return path


def test_train_dense_depth_probe_smoke_end_to_end(tmp_path: Path) -> None:
    manifest = _build_synthetic_manifest(tmp_path)
    manifest_path = _write_min_manifest_parquet(
        tmp_path / "manifests" / "render_valid.parquet", manifest
    )

    cache_dir = tmp_path / "features"
    cache_path = patch_feature_cache_path(
        cache_dir, model_name="m", layer_name="final"
    )
    feature_dim = 4
    patch_side = 4
    # Build features that correlate with the depth target so the probe can fit.
    render_ids = manifest["render_id"].tolist()
    patches = np.zeros(
        (len(render_ids), patch_side, patch_side, feature_dim), dtype=np.float32
    )
    targets_for_check = []
    for i, rid in enumerate(render_ids):
        row = manifest[manifest["render_id"] == rid].iloc[0]
        depth = np.load(row["depth_path"]).astype(np.float32)
        mask = np.load(row["mask_path"]).astype(bool)
        from exp1.tasks.dense_depth import nanaware_pool

        pooled, valid = nanaware_pool(
            depth, grid_rows=patch_side, grid_cols=patch_side, mask=mask
        )
        targets_for_check.append(pooled)
        for r in range(patch_side):
            for c in range(patch_side):
                if not valid[r, c]:
                    continue
                base = float(pooled[r, c])
                patches[i, r, c, 0] = base
                patches[i, r, c, 1] = base * 0.5 + 0.1
                patches[i, r, c, 2] = base * 0.25
                patches[i, r, c, 3] = 1.0
    save_patch_feature_cache(
        cache_path,
        render_ids=render_ids,
        patch_features=patches,
        metadata={"layer_name": "final"},
    )

    output_dir = tmp_path / "outputs" / "dense"
    result = train_dense_depth_probe(
        patch_cache=cache_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_name="m",
        layer_name="final",
        texture_condition=["photorealistic"],
        config=DenseDepthTrainConfig(
            epochs=20,
            batch_size=2,
            lr=5e-2,
            weight_decay=0.0,
            device="cpu",
            use_layernorm=False,
            scheduler="none",
            early_stopping=False,
        ),
    )

    assert "train" in result["metrics"]
    assert "test" in result["metrics"]
    train_metrics = result["metrics"]["train"]
    assert train_metrics["pearson_r_mean"] > 0.8
    metrics_path = result["artifact_paths"]["metrics"]
    assert Path(metrics_path).is_file()
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    assert payload["metadata"]["task"] == result["train_config"]["task"]
    assert payload["metadata"]["feature_mode"] == "patch"
    train_predictions = output_dir / "predictions_train.npz"
    assert train_predictions.is_file()
    assert (output_dir / "predictions.csv").is_file()


def test_patch_cache_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "model" / "layer_patch.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ids = ["a", "b", "c"]
    features = np.random.RandomState(0).randn(3, 4, 4, 8).astype(np.float16)
    save_patch_feature_cache(
        cache_path,
        render_ids=ids,
        patch_features=features,
        metadata={"hello": "world"},
    )
    loaded = load_patch_feature_cache(cache_path)
    try:
        np.testing.assert_array_equal(loaded["render_ids"], np.asarray(ids))
        assert loaded["patch_features"].shape == (3, 4, 4, 8)
        assert loaded["cls_features"] is None
        assert loaded["patch_grid_shape"] == (4, 4)
        assert loaded["feature_dim"] == 8
        assert loaded["metadata"]["hello"] == "world"
    finally:
        loaded["_npz_handle"].close()


def test_patch_cache_roundtrip_with_cls_features(tmp_path: Path) -> None:
    cache_path = tmp_path / "model" / "layer_patch_cls.npz"
    ids = ["a", "b"]
    features = np.random.RandomState(1).randn(2, 3, 3, 4).astype(np.float32)
    cls = np.random.RandomState(2).randn(2, 5).astype(np.float32)
    save_patch_feature_cache(
        cache_path,
        render_ids=ids,
        patch_features=features,
        cls_features=cls,
        metadata={"model_input_size": 224, "patch_size": 16},
    )
    loaded = load_patch_feature_cache(cache_path)
    try:
        assert loaded["cls_features"].shape == (2, 5)
        assert loaded["metadata"]["model_input_size"] == 224
    finally:
        loaded["_npz_handle"].close()


def test_dense_depth_dataset_can_concatenate_patch_and_cls_features(
    tmp_path: Path,
) -> None:
    manifest = _build_synthetic_manifest(tmp_path).iloc[:1].copy()
    cache_path = patch_feature_cache_path(
        tmp_path / "features", model_name="m", layer_name="final"
    )
    save_patch_feature_cache(
        cache_path,
        render_ids=manifest["render_id"].tolist(),
        patch_features=np.zeros((1, 2, 2, 3), dtype=np.float32),
        cls_features=np.ones((1, 5), dtype=np.float32),
    )
    dataset = Exp1DenseDepthDataset(
        cache_path,
        manifest=manifest,
        feature_mode="patch_cls",
        depth_statistic="mean",
        min_valid_fraction_per_patch=0.0,
    )
    sample = dataset[0]
    assert sample.features.shape == (2, 2, 8)
    np.testing.assert_allclose(sample.features[..., 3:], 1.0)


def test_align_depth_mask_to_model_input_resizes_and_center_crops() -> None:
    depth = np.arange(8, dtype=np.float32).reshape(2, 4)
    mask = np.ones_like(depth, dtype=bool)
    aligned_depth, aligned_mask = align_depth_mask_to_model_input(
        depth,
        mask,
        model_input_size=4,
    )
    assert aligned_depth.shape == (4, 4)
    assert aligned_mask.shape == (4, 4)

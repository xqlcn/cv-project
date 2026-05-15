"""Unit tests for dense per-patch surface-normal probes."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from exp1.features.storage import patch_feature_cache_path, save_patch_feature_cache
from exp1.probes.dense_surface_normals import (
    DenseSurfaceNormalHead,
    DenseSurfaceNormalHeadConfig,
    dense_surface_normal_loss,
    dense_surface_normal_metrics,
)
from exp1.probes.train_dense_surface_normals import (
    DenseSurfaceNormalTrainConfig,
    train_dense_surface_normal_probe,
)
from exp1.tasks.dense_surface_normals import (
    Exp1DenseSurfaceNormalDataset,
    align_normal_mask_to_model_input,
    pool_surface_normals,
)


def _normal_field(image_size: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(-0.7, 0.7, image_size, dtype=np.float32),
        np.linspace(-0.7, 0.7, image_size, dtype=np.float32),
        indexing="ij",
    )
    normals = np.stack(
        [
            xx + rng.normal(0.0, 0.02, size=xx.shape).astype(np.float32),
            yy + rng.normal(0.0, 0.02, size=yy.shape).astype(np.float32),
            np.ones_like(xx),
        ],
        axis=-1,
    )
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
    return normals.astype(np.float32)


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
    mask = np.zeros((image_size, image_size), dtype=bool)
    mask[2 : image_size - 2, 2 : image_size - 2] = True
    depth = np.ones((image_size, image_size), dtype=np.float32)
    depth[~mask] = np.nan
    normal = _normal_field(image_size, seed=seed)
    normal[~mask] = np.nan
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
    for split, obj_id in (
        ("train", "objA"),
        ("train", "objB"),
        ("val", "objC"),
        ("test", "objD"),
    ):
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


def test_pool_surface_normals_normalizes_patch_means() -> None:
    normals = np.zeros((4, 4, 3), dtype=np.float32)
    normals[..., 0] = 0.5
    normals[..., 2] = 1.0
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    mask = np.ones((4, 4), dtype=bool)

    pooled, valid = pool_surface_normals(normals, grid_rows=2, grid_cols=2, mask=mask)

    assert pooled.shape == (2, 2, 3)
    assert valid.all()
    np.testing.assert_allclose(np.linalg.norm(pooled, axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(pooled[0, 0], normals[0, 0], atol=1e-6)


def test_dense_surface_normal_head_loss_and_metrics() -> None:
    head = DenseSurfaceNormalHead(DenseSurfaceNormalHeadConfig(feature_dim=5))
    features = torch.randn(2, 3, 3, 5)
    output = head(features)
    assert tuple(output.shape) == (2, 3, 3, 3)

    target = torch.zeros(2, 3, 3, 3)
    target[..., 2] = 1.0
    pred = target.clone()
    mask = torch.ones(2, 3, 3, dtype=torch.bool)
    loss = dense_surface_normal_loss(pred, target, mask)
    metrics = dense_surface_normal_metrics(pred, target, mask)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert metrics["angular_error_deg_mean"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["within_30_deg"] == pytest.approx(1.0)


def test_dense_surface_normal_dataset_pools_targets(tmp_path: Path) -> None:
    manifest = _build_synthetic_manifest(tmp_path)
    cache_path = patch_feature_cache_path(tmp_path / "features", model_name="m", layer_name="final")
    patch_side = 4
    feature_dim = 6
    render_ids = manifest["render_id"].tolist()
    patches = np.random.RandomState(0).randn(
        len(render_ids),
        patch_side,
        patch_side,
        feature_dim,
    ).astype(np.float32)
    save_patch_feature_cache(
        cache_path,
        render_ids=render_ids,
        patch_features=patches,
        metadata={"layer_name": "final"},
    )

    dataset = Exp1DenseSurfaceNormalDataset(
        cache_path,
        manifest=manifest,
        split="train",
        texture_condition=["photorealistic"],
    )
    sample = dataset[0]
    assert len(dataset) == 2
    assert sample.features.shape == (patch_side, patch_side, feature_dim)
    assert sample.target.shape == (patch_side, patch_side, 3)
    assert sample.valid.dtype == bool


def test_align_normal_mask_to_model_input_resizes_and_center_crops() -> None:
    normal = np.zeros((2, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    mask = np.ones((2, 4), dtype=bool)
    aligned_normal, aligned_mask = align_normal_mask_to_model_input(
        normal,
        mask,
        model_input_size=4,
    )
    assert aligned_normal.shape == (4, 4, 3)
    assert aligned_mask.shape == (4, 4)


def _write_min_manifest_parquet(path: Path, manifest: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)
    return path


def test_train_dense_surface_normal_probe_smoke_end_to_end(tmp_path: Path) -> None:
    manifest = _build_synthetic_manifest(tmp_path)
    manifest_path = _write_min_manifest_parquet(
        tmp_path / "manifests" / "render_valid.parquet",
        manifest,
    )
    cache_path = patch_feature_cache_path(tmp_path / "features", model_name="m", layer_name="final")
    patch_side = 4
    feature_dim = 5
    render_ids = manifest["render_id"].tolist()
    patches = np.zeros(
        (len(render_ids), patch_side, patch_side, feature_dim),
        dtype=np.float32,
    )
    for i, rid in enumerate(render_ids):
        row = manifest[manifest["render_id"] == rid].iloc[0]
        normals = np.load(row["normal_path"]).astype(np.float32)
        mask = np.load(row["mask_path"]).astype(bool)
        pooled, valid = pool_surface_normals(
            normals,
            grid_rows=patch_side,
            grid_cols=patch_side,
            mask=mask,
        )
        patches[i, ..., :3] = pooled
        patches[i, ..., 3] = valid.astype(np.float32)
        patches[i, ..., 4] = 1.0
    save_patch_feature_cache(
        cache_path,
        render_ids=render_ids,
        patch_features=patches,
        metadata={"layer_name": "final"},
    )

    output_dir = tmp_path / "outputs" / "dense_normals"
    result = train_dense_surface_normal_probe(
        patch_cache=cache_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        model_name="m",
        layer_name="final",
        texture_condition=["photorealistic"],
        config=DenseSurfaceNormalTrainConfig(
            epochs=25,
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
    assert result["metrics"]["train"]["angular_error_deg_mean"] < 20.0
    metrics_path = result["artifact_paths"]["metrics"]
    assert Path(metrics_path).is_file()
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    assert payload["metadata"]["task"] == "dense_surface_normal_patches"
    assert (output_dir / "predictions_train.npz").is_file()
    assert (output_dir / "predictions.csv").is_file()

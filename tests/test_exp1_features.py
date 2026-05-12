from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from exp1.features.extract import extract_feature_arrays, parse_layer_name
from exp1.features.storage import load_feature_cache, save_feature_cache


def test_parse_layer_name_accepts_final_and_intermediate_layers() -> None:
    assert parse_layer_name("final") is None
    assert parse_layer_name("layer4") == 4


def test_extract_feature_arrays_are_finite_deterministic_and_normalized(
    tmp_path,
) -> None:
    image_paths = []
    for idx, color in enumerate([(255, 0, 0), (0, 128, 0), (0, 0, 64)]):
        path = tmp_path / f"image_{idx}.png"
        Image.new("RGB", (8, 8), color=color).save(path)
        image_paths.append(path)
    rows = [
        {"render_id": f"r{idx}", "rgb_path": str(path)}
        for idx, path in enumerate(image_paths)
    ]

    def batch_extract(images):
        values = []
        for image in images:
            arr = np.asarray(image, dtype=np.float32) / 255.0
            values.append(arr.mean(axis=(0, 1)))
        cls = torch.tensor(np.stack(values, axis=0), dtype=torch.float32)
        return {
            "cls_final": cls,
            "patch_tokens_final": cls[:, None, :],
            "layer_cls": {4: cls + 1.0},
        }

    first = extract_feature_arrays(
        rows,
        batch_extract_fn=batch_extract,
        layer_names=["final", "layer4"],
        batch_size=2,
        normalize=True,
        show_progress=False,
    )
    second = extract_feature_arrays(
        rows,
        batch_extract_fn=batch_extract,
        layer_names=["final", "layer4"],
        batch_size=2,
        normalize=True,
        show_progress=False,
    )

    assert set(first) == {"final", "layer4"}
    assert first["final"].shape == (3, 3)
    assert np.isfinite(first["layer4"]).all()
    assert np.allclose(np.linalg.norm(first["final"], axis=1), 1.0)
    assert np.allclose(first["final"], second["final"])


def test_feature_cache_storage_round_trip_includes_metadata(tmp_path) -> None:
    path = tmp_path / "clip_vit_b16" / "final.npz"
    save_feature_cache(
        path,
        render_ids=["r1", "r2"],
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        metadata={"model_name": "clip_vit_b16", "layer_name": "final"},
    )

    loaded = load_feature_cache(path)

    assert loaded["render_ids"].tolist() == ["r1", "r2"]
    assert loaded["features"].dtype == np.float32
    assert loaded["metadata"]["model_name"] == "clip_vit_b16"

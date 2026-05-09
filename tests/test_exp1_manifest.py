from __future__ import annotations

import pandas as pd
import pytest

from exp1.metadata.manifest import (
    ManifestValidationError,
    attach_render_ids,
    load_manifest,
    save_manifest,
    validate_render_manifest,
)
from exp1.metadata.schema import REQUIRED_RENDER_COLUMNS, generate_render_id


def _base_record(**overrides):
    record = {
        "object_id": "obj_001",
        "source_dataset": "synthetic_primitives",
        "category": "chair",
        "split": "train",
        "raw_mesh_path": "data/raw/obj_001.obj",
        "normalized_mesh_path": "data/normalized/obj_001.glb",
        "texture_condition": "flat",
        "texture_seed": 11,
        "camera_distance": 2.4,
        "camera_azimuth_deg": 0.0,
        "camera_elevation_deg": 20.0,
        "camera_fov_deg": 50.0,
        "object_scale": 1.0,
        "light_type": "sun",
        "light_azimuth_deg": 45.0,
        "light_elevation_deg": 35.0,
        "light_intensity": 3.0,
        "rgb_path": "data/exp1/renders/rgb.png",
        "depth_path": "data/exp1/renders/depth.npy",
        "normal_path": "data/exp1/renders/normal_camera.npy",
        "mask_path": "data/exp1/renders/mask.npy",
        "render_seed": 7,
        "render_status": "pending",
        "qc_error_message": "",
    }
    record.update(overrides)
    record["render_id"] = generate_render_id(record)
    return record


def test_generate_render_id_is_deterministic_and_path_independent() -> None:
    record = _base_record()
    shuffled = dict(reversed(list(record.items())))

    assert generate_render_id(record) == generate_render_id(shuffled)

    moved = dict(record)
    moved["rgb_path"] = "/tmp/somewhere-else/rgb.png"
    moved["render_status"] = "success"
    assert generate_render_id(record) == generate_render_id(moved)


def test_generate_render_id_changes_for_controlled_fields() -> None:
    first = _base_record(texture_condition="flat")
    second = _base_record(texture_condition="random_noise")

    assert generate_render_id(first) != generate_render_id(second)


def test_generate_render_id_is_stable_for_dataframe_records() -> None:
    record = _base_record()
    dataframe_record = pd.DataFrame([record]).to_dict(orient="records")[0]

    assert generate_render_id(record) == generate_render_id(dataframe_record)


def test_attach_render_ids_fills_missing_ids() -> None:
    record = _base_record()
    record["render_id"] = ""

    df = attach_render_ids([record])

    assert df.loc[0, "render_id"] == generate_render_id(record)


def test_validate_render_manifest_accepts_valid_records() -> None:
    df = validate_render_manifest([_base_record()])

    assert list(df.columns)
    assert set(REQUIRED_RENDER_COLUMNS).issubset(df.columns)


def test_validate_render_manifest_reports_missing_columns() -> None:
    record = _base_record()
    del record["depth_path"]

    with pytest.raises(ManifestValidationError, match="Missing required.*depth_path"):
        validate_render_manifest([record])


def test_validate_render_manifest_reports_invalid_split() -> None:
    record = _base_record(split="dev")

    with pytest.raises(ManifestValidationError, match="Invalid split.*dev"):
        validate_render_manifest([record])


def test_validate_render_manifest_reports_null_split() -> None:
    record = _base_record(split=None)

    with pytest.raises(ManifestValidationError, match="split.*null"):
        validate_render_manifest([record])


def test_validate_render_manifest_reports_invalid_texture_condition() -> None:
    record = _base_record(texture_condition="noise")

    with pytest.raises(ManifestValidationError, match="Invalid texture.*noise"):
        validate_render_manifest([record])


def test_validate_render_manifest_reports_null_texture_condition() -> None:
    record = _base_record(texture_condition=None)

    with pytest.raises(ManifestValidationError, match="texture_condition.*null"):
        validate_render_manifest([record])


def test_validate_render_manifest_reports_duplicate_render_ids() -> None:
    first = _base_record(object_id="obj_001")
    second = _base_record(object_id="obj_002")
    second["render_id"] = first["render_id"]

    with pytest.raises(ManifestValidationError, match="Duplicate render_id"):
        validate_render_manifest([first, second])


def test_save_and_load_jsonl_manifest_round_trip(tmp_path) -> None:
    rows = [
        _base_record(object_id="obj_001", split="train"),
        _base_record(object_id="obj_002", split="val"),
    ]
    path = tmp_path / "render_manifest.jsonl"

    save_manifest(rows, path)
    loaded = load_manifest(path)

    assert loaded["render_id"].tolist() == [row["render_id"] for row in rows]
    assert loaded["split"].tolist() == ["train", "val"]


def test_save_and_load_csv_manifest_round_trip(tmp_path) -> None:
    row = _base_record(object_id="obj_001")
    path = tmp_path / "render_manifest.csv"

    save_manifest(pd.DataFrame([row]), path)
    loaded = load_manifest(path)

    assert loaded.loc[0, "render_id"] == row["render_id"]

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from exp1.assets.discover import (
    discover_assets_from_directory,
    discover_modelnet40_assets,
)
from exp1.assets.normalize import normalize_asset_manifest
from exp1.assets.validate import (
    AssetValidationError,
    build_object_split_manifest,
    validate_asset_manifest,
    write_object_split_manifest,
)
from src.utils.io import read_jsonl


def _write_box(
    path: Path,
    *,
    extents=(2.0, 4.0, 6.0),
    translation=(5.0, 0.0, -3.0),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(translation)
    mesh.export(path)


def test_discover_assets_from_directory_finds_supported_meshes(tmp_path) -> None:
    mesh_path = tmp_path / "train" / "chair" / "chair_0001.obj"
    _write_box(mesh_path)
    ignored = tmp_path / "train" / "chair" / "notes.txt"
    ignored.write_text("not a mesh", encoding="utf-8")

    rows = discover_assets_from_directory(tmp_path, source_dataset="unit")

    assert len(rows) == 1
    assert rows[0]["object_id"] == "chair_0001"
    assert rows[0]["source_dataset"] == "unit"
    assert rows[0]["category"] == "chair"
    assert rows[0]["split"] == "train"
    assert rows[0]["raw_mesh_path"] == str(mesh_path.resolve())


def test_discover_modelnet40_assets_uses_existing_repo_scanner(tmp_path) -> None:
    mesh_path = tmp_path / "train" / "chair" / "chair_0001.off"
    _write_box(mesh_path)

    rows = discover_modelnet40_assets(tmp_path)

    assert len(rows) == 1
    assert rows[0]["object_id"] == "chair_chair_0001"
    assert rows[0]["source_dataset"] == "modelnet40"
    assert rows[0]["split"] == "train"
    assert rows[0]["raw_mesh_path"] == str(mesh_path.resolve())


def test_normalize_asset_manifest_writes_centered_unit_glb(tmp_path) -> None:
    raw = tmp_path / "raw" / "train" / "box" / "box_0001.obj"
    _write_box(raw)
    rows = [
        {
            "object_id": "box_0001",
            "source_dataset": "unit",
            "category": "box",
            "split": "train",
            "raw_mesh_path": str(raw),
            "normalized_mesh_path": "",
            "asset_status": "discovered",
            "asset_error_message": "",
        }
    ]

    normalized = normalize_asset_manifest(rows, output_root=tmp_path / "normalized")
    validate_asset_manifest(normalized)

    out_path = Path(normalized[0]["normalized_mesh_path"])
    mesh = trimesh.load(out_path, force="mesh")
    bounds = np.asarray(mesh.bounds)
    center = (bounds[0] + bounds[1]) / 2.0
    max_extent = float((bounds[1] - bounds[0]).max())

    assert out_path.suffix == ".glb"
    assert np.allclose(center, np.zeros(3), atol=1e-6)
    assert max_extent == pytest.approx(1.0, abs=1e-6)
    assert normalized[0]["asset_status"] == "normalized"


def test_invalid_mesh_is_marked_without_crashing(tmp_path) -> None:
    raw = tmp_path / "bad.obj"
    raw.write_text("this is not a mesh", encoding="utf-8")
    rows = [
        {
            "object_id": "bad",
            "source_dataset": "unit",
            "category": "bad",
            "split": "train",
            "raw_mesh_path": str(raw),
            "normalized_mesh_path": "",
            "asset_status": "discovered",
            "asset_error_message": "",
        }
    ]

    normalized = normalize_asset_manifest(rows, output_root=tmp_path / "normalized")
    validate_asset_manifest(normalized)

    assert normalized[0]["asset_status"] == "failed"
    assert normalized[0]["asset_error_message"]
    assert normalized[0]["normalized_mesh_path"] == ""


def test_object_split_manifest_is_disjoint(tmp_path) -> None:
    rows = [
        {
            "object_id": "a",
            "source_dataset": "unit",
            "category": "box",
            "split": "train",
        },
        {
            "object_id": "b",
            "source_dataset": "unit",
            "category": "box",
            "split": "val",
        },
    ]

    split_rows = build_object_split_manifest(rows)
    split_path = write_object_split_manifest(rows, tmp_path / "splits.jsonl")

    assert {row["object_id"] for row in split_rows} == {"a", "b"}
    assert split_path.is_file()
    assert read_jsonl(split_path) == split_rows


def test_object_split_manifest_rejects_leakage() -> None:
    rows = [
        {
            "object_id": "leaky",
            "source_dataset": "unit",
            "category": "box",
            "split": "train",
        },
        {
            "object_id": "leaky",
            "source_dataset": "unit",
            "category": "box",
            "split": "test",
        },
    ]

    with pytest.raises(AssetValidationError, match="multiple splits"):
        build_object_split_manifest(rows)


def test_preprocess_assets_script_supports_modelnet40(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mesh_path = tmp_path / "modelnet40" / "train" / "chair" / "chair_0001.off"
    _write_box(mesh_path)
    asset_manifest = tmp_path / "assets.jsonl"
    normalized_manifest = tmp_path / "assets_normalized.jsonl"
    split_manifest = tmp_path / "splits.jsonl"
    normalized_root = tmp_path / "normalized"

    subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_assets.py",
            "--config",
            "configs/exp1_smoke.yaml",
            "--modelnet-root",
            str(tmp_path / "modelnet40"),
            "--output-manifest",
            str(asset_manifest),
            "--normalized-manifest",
            str(normalized_manifest),
            "--split-manifest",
            str(split_manifest),
            "--normalized-root",
            str(normalized_root),
        ],
        cwd=repo_root,
        check=True,
    )

    raw_rows = read_jsonl(asset_manifest)
    normalized_rows = read_jsonl(normalized_manifest)
    split_rows = read_jsonl(split_manifest)

    assert len(raw_rows) == 1
    assert len(normalized_rows) == 1
    assert normalized_rows[0]["source_dataset"] == "modelnet40"
    assert normalized_rows[0]["asset_status"] == "normalized"
    assert Path(normalized_rows[0]["normalized_mesh_path"]).is_file()
    assert split_rows == [
        {
            "object_id": "chair_chair_0001",
            "source_dataset": "modelnet40",
            "category": "chair",
            "split": "train",
        }
    ]

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exp1.metadata.manifest import load_manifest
from exp1.metadata.schema import VALID_TEXTURE_CONDITIONS
from exp1.rendering.grid import build_render_plan, load_asset_manifest


CONTROL_FIELDS = [
    "object_id",
    "camera_distance",
    "camera_azimuth_deg",
    "camera_elevation_deg",
    "camera_fov_deg",
    "object_scale",
    "light_type",
    "light_azimuth_deg",
    "light_elevation_deg",
    "light_intensity",
    "render_seed",
]


def _build_plan(asset_rows, tmp_path: Path):
    return build_render_plan(
        asset_rows,
        texture_conditions=VALID_TEXTURE_CONDITIONS,
        camera_distances=[2.4],
        camera_azimuths_deg=[0.0, 90.0],
        camera_elevations_deg=[20.0],
        camera_fov_deg=50.0,
        light_type="sun",
        light_azimuths_deg=[45.0],
        light_elevations_deg=[35.0],
        light_intensities=[3.0],
        object_scales=[1.0],
        render_root=tmp_path / "renders",
        seed=7,
    )


def test_every_object_appears_in_all_texture_conditions(tmp_path) -> None:
    rows = [
        {
            "object_id": "syn_box_0001",
            "dataset": "synthetic_primitives",
            "category": "box",
            "split": "train",
            "mesh_path": "data/synthetic_primitives/train/syn_box_0001.obj",
        },
        {
            "object_id": "syn_cone_0002",
            "dataset": "synthetic_primitives",
            "category": "cone",
            "split": "val",
            "mesh_path": "data/synthetic_primitives/val/syn_cone_0002.obj",
        },
    ]

    plan = _build_plan(rows, tmp_path)

    for object_id, group in plan.groupby("object_id"):
        assert object_id in {"syn_box_0001", "syn_cone_0002"}
        assert set(group["texture_condition"]) == set(VALID_TEXTURE_CONDITIONS)


def test_texture_only_controls_match_camera_lighting_and_scale(tmp_path) -> None:
    plan = _build_plan(
        [
            {
                "object_id": "syn_box_0001",
                "source_dataset": "synthetic_primitives",
                "category": "box",
                "split": "train",
                "raw_mesh_path": "data/raw/syn_box_0001.obj",
                "normalized_mesh_path": "data/normalized/syn_box_0001.glb",
            }
        ],
        tmp_path,
    )

    for _, group in plan.groupby("texture_control_group_id"):
        assert set(group["texture_condition"]) == set(VALID_TEXTURE_CONDITIONS)
        assert len(group) == len(VALID_TEXTURE_CONDITIONS)
        for field in CONTROL_FIELDS:
            assert group[field].nunique() == 1


def test_render_ids_are_unique(tmp_path) -> None:
    plan = _build_plan(
        [
            {
                "object_id": "syn_box_0001",
                "dataset": "synthetic_primitives",
                "category": "box",
                "split": "train",
                "mesh_path": "data/synthetic_primitives/train/syn_box_0001.obj",
            }
        ],
        tmp_path,
    )

    assert plan["render_id"].is_unique


def test_split_labels_are_preserved(tmp_path) -> None:
    rows = [
        {
            "object_id": "syn_box_0001",
            "dataset": "synthetic_primitives",
            "category": "box",
            "split": "train",
            "mesh_path": "data/synthetic_primitives/train/syn_box_0001.obj",
        },
        {
            "object_id": "syn_sphere_0002",
            "dataset": "synthetic_primitives",
            "category": "sphere",
            "split": "test",
            "mesh_path": "data/synthetic_primitives/test/syn_sphere_0002.obj",
        },
    ]

    plan = _build_plan(rows, tmp_path)

    observed = plan.groupby("object_id")["split"].unique().to_dict()
    assert list(observed["syn_box_0001"]) == ["train"]
    assert list(observed["syn_sphere_0002"]) == ["test"]


def test_modelnet40_rows_are_supported_and_split_is_inferred(tmp_path) -> None:
    train_mesh = tmp_path / "modelnet40" / "train" / "chair" / "chair_0001.off"
    test_mesh = tmp_path / "modelnet40" / "test" / "chair" / "chair_0002.off"
    rows = [
        {
            "object_id": "chair_chair_0001",
            "dataset": "modelnet40",
            "category": "chair",
            "mesh_path": str(train_mesh),
        },
        {
            "object_id": "chair_chair_0002",
            "dataset": "modelnet40",
            "category": "chair",
            "mesh_path": str(test_mesh),
        },
    ]

    plan = _build_plan(rows, tmp_path)

    by_object = plan.groupby("object_id").first()
    assert by_object.loc["chair_chair_0001", "split"] == "train"
    assert by_object.loc["chair_chair_0002", "split"] == "test"
    assert set(plan["source_dataset"]) == {"modelnet40"}
    assert by_object.loc["chair_chair_0001", "raw_mesh_path"] == str(train_mesh)
    assert by_object.loc["chair_chair_0001", "normalized_mesh_path"] == str(train_mesh)


def test_asset_material_metadata_is_preserved_for_render_plan(tmp_path) -> None:
    plan = _build_plan(
        [
            {
                "object_id": "chair_abc123",
                "source_dataset": "shapenet",
                "category": "chair",
                "split": "train",
                "raw_mesh_path": (
                    "data/shapenet/03001627/abc123/models/model_normalized.obj"
                ),
                "normalized_mesh_path": "data/normalized/chair_abc123.glb",
                "has_photorealistic_material": True,
                "hf_repo_id": "ShapeNet/ShapeNetCore",
                "shapenet_synset_id": "03001627",
                "shapenet_model_id": "abc123",
            }
        ],
        tmp_path,
    )

    first = plan.iloc[0]
    assert bool(first["has_photorealistic_material"]) is True
    assert first["hf_repo_id"] == "ShapeNet/ShapeNetCore"
    assert first["shapenet_synset_id"] == "03001627"
    assert first["shapenet_model_id"] == "abc123"


def test_load_asset_manifest_accepts_modelnet_json(tmp_path) -> None:
    manifest_path = tmp_path / "modelnet40_manifest.json"
    rows = [
        {
            "object_id": "chair_chair_0001",
            "category": "chair",
            "mesh_path": "data/modelnet40/train/chair/chair_0001.off",
            "dataset": "modelnet40",
        }
    ]
    manifest_path.write_text(json.dumps(rows), encoding="utf-8")

    loaded = load_asset_manifest(manifest_path)

    assert loaded == rows


def test_create_render_plan_script_writes_jsonl(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    asset_manifest = tmp_path / "assets.json"
    output = tmp_path / "render_plan.jsonl"
    rows = [
        {
            "object_id": "chair_chair_0001",
            "category": "chair",
            "mesh_path": "data/modelnet40/train/chair/chair_0001.off",
            "dataset": "modelnet40",
        }
    ]
    asset_manifest.write_text(json.dumps(rows), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/create_render_plan.py",
            "--config",
            "configs/exp1_smoke.yaml",
            "--asset-manifest",
            str(asset_manifest),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=True,
    )
    plan = load_manifest(output)

    assert len(plan) == 6
    assert set(plan["texture_condition"]) == set(VALID_TEXTURE_CONDITIONS)
    assert plan["render_id"].is_unique

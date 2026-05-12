from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import SimpleMaterial
from trimesh.visual.texture import TextureVisuals


def _write_mesh(path: Path, *, textured: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
    mesh.apply_translation((3.0, 0.0, -2.0))
    if textured:
        vertices = np.asarray(mesh.vertices)
        uv = (vertices[:, :2] - vertices[:, :2].min(axis=0)) / np.maximum(
            np.ptp(vertices[:, :2], axis=0),
            1e-8,
        )
        noise = np.random.default_rng(123).integers(
            0,
            256,
            size=(16, 16, 3),
            dtype=np.uint8,
        )
        mesh.visual = TextureVisuals(
            uv=uv,
            material=SimpleMaterial(
                image=Image.fromarray(noise),
                diffuse=[255, 255, 255, 255],
            ),
        )
    mesh.export(path)


def test_sanity_check_preprocessing_writes_and_validates_texture_glbs(
    tmp_path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    shapenet_root = tmp_path / "shapenetcore"
    objaverse_root = tmp_path / "objaverse"
    output_dir = tmp_path / "outputs"

    _write_mesh(
        shapenet_root / "03001627" / "abc123" / "models" / "model_normalized.obj",
        textured=True,
    )
    _write_mesh(
        shapenet_root / "04379243" / "def456" / "models" / "model_normalized.obj",
        textured=True,
    )
    _write_mesh(
        objaverse_root / "train" / "chair" / "objaverse_chair.glb",
        textured=True,
    )
    _write_mesh(
        objaverse_root / "train" / "table" / "objaverse_table.glb",
        textured=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/sanity_check_preprocessing.py",
            "--shapenetcore-root",
            str(shapenet_root),
            "--objaverse-root",
            str(objaverse_root),
            "--output-dir",
            str(output_dir),
            "--num-objects-per-dataset",
            "2",
            "--texture-conditions",
            "photorealistic",
            "flat",
            "random_noise",
            "--seed",
            "11",
            "--strict",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report_path = output_dir / "metadata" / "preprocessing_report.json"
    markdown_path = output_dir / "metadata" / "preprocessing_report.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert markdown_path.is_file()
    assert report["summary"]["num_shapenetcore_objects_processed"] == 2
    assert report["summary"]["num_objaverse_objects_processed"] == 2
    assert report["summary"]["num_texture_condition_variants_written"] == 12
    assert report["summary"]["num_glb_files_written"] == 12
    assert report["summary"]["num_failures"] == 0

    artifacts = report["artifacts"]
    assert len(artifacts) == 12
    for artifact in artifacts:
        path = Path(artifact["output_glb_path"])
        assert path.is_file()
        assert path.stat().st_size > 0
        assert artifact["validation"]["inspection"]["reload_ok"] is True
        assert artifact["validation"]["inspection"]["has_geometry"] is True

    flat_artifacts = [a for a in artifacts if a["texture_condition"] == "flat"]
    assert flat_artifacts
    for artifact in flat_artifacts:
        inspection = artifact["validation"]["inspection"]
        assert inspection["has_material"] is True
        assert inspection["has_texture"] is False
        assert inspection["material_base_colors"][0][:3] == [158, 158, 158]

    noise_artifacts = [a for a in artifacts if a["texture_condition"] == "random_noise"]
    assert noise_artifacts
    for artifact in noise_artifacts:
        inspection = artifact["validation"]["inspection"]
        assert inspection["has_texture"] is True
        assert inspection["texture_image_sha1"]

    shapenet_photo = [
        a
        for a in artifacts
        if a["source_dataset"] == "shapenetcore"
        and a["texture_condition"] == "photorealistic"
    ]
    assert {
        a["material_texture_metadata"]["material_status"] for a in shapenet_photo
    } == {"preserved_original"}

    objaverse_photo = [
        a
        for a in artifacts
        if a["source_dataset"] == "objaverse"
        and a["texture_condition"] == "photorealistic"
    ]
    assert {
        a["material_texture_metadata"]["material_status"] for a in objaverse_photo
    } == {"preserved_original"}

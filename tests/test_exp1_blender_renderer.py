from __future__ import annotations

import builtins
import importlib.util
import json
import py_compile
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from exp1.rendering.materials import (
    apply_materials,
    material_mode,
    photorealistic_fallback_reason,
)
from scripts import render_blender
from src.utils.io import read_jsonl


def _load_material_utils_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "blender" / "material_utils.py"
    spec = importlib.util.spec_from_file_location("material_utils_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_blender_script_compiles() -> None:
    py_compile.compile(
        "scripts/render_blender.py",
        cfile="/tmp/render_blender_test.pyc",
        doraise=True,
    )


def test_render_blender_help_works_without_blender() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/render_blender.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--chunk" in result.stdout


def test_blender_material_import_does_not_require_pandas() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import builtins
real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "pandas":
        raise ModuleNotFoundError("No module named 'pandas'")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
from scripts import render_blender
assert "flat" in render_blender.DEFAULT_CONFIG["textures"]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
    )


def test_render_blender_config_loads_without_pyyaml(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    cfg = render_blender._load_config(
        Path("configs/exp1_smoke.yaml"),
        Path.cwd(),
    )

    assert cfg["render"]["resolution"] == [128, 128]
    assert cfg["render"]["samples"] == 16
    assert cfg["textures"]["flat"]["color_rgb"] == [0.62, 0.62, 0.62]


def test_random_noise_material_mode_maps_to_existing_helper() -> None:
    assert material_mode("random_noise") == "image_noise"
    assert material_mode("flat") == "flat"


def _install_fake_material_utils(monkeypatch):
    calls = []

    def assign_principled_material(obj, **kwargs):
        calls.append({"object": obj, **kwargs})
        obj.assigned_material = kwargs

    def has_preservable_material(obj):
        return bool(getattr(obj, "preservable_material", False))

    fake_module = types.SimpleNamespace(
        assign_principled_material=assign_principled_material,
        has_preservable_material=has_preservable_material,
    )
    monkeypatch.setitem(sys.modules, "material_utils", fake_module)
    return calls


class _FakeObject:
    def __init__(self, *, preservable_material: bool = False) -> None:
        self.preservable_material = preservable_material


def test_textureless_photorealistic_materials_fallback(monkeypatch) -> None:
    calls = _install_fake_material_utils(monkeypatch)
    record = {
        "texture_condition": "photorealistic",
        "texture_seed": 123,
        "source_dataset": "modelnet40",
        "raw_mesh_path": "chair/train/chair_0001.off",
    }

    meta = apply_materials([_FakeObject(preservable_material=True)], record, {})

    assert meta["material_status"] == "fallback"
    assert meta["photorealistic_material_status"] == "fallback_missing_original"
    assert "raw_mesh_extension_textureless" in meta["photorealistic_fallback_reason"]
    assert calls[0]["texture_type"] == "photorealistic"
    assert calls[0]["preserve_existing"] is False


def test_photorealistic_materials_preserve_imported_non_modelnet(monkeypatch) -> None:
    calls = _install_fake_material_utils(monkeypatch)
    record = {
        "texture_condition": "photorealistic",
        "texture_seed": 123,
        "source_dataset": "objaverse",
        "raw_mesh_path": "asset.glb",
    }

    meta = apply_materials([_FakeObject(preservable_material=True)], record, {})

    assert meta["material_status"] == "preserved"
    assert meta["photorealistic_material_status"] == "preserved"
    assert calls == []


def test_photorealistic_fallback_reason_honors_manifest_flags() -> None:
    reason = photorealistic_fallback_reason(
        {"has_photorealistic_material": False, "source_dataset": "objaverse"},
        {},
    )

    assert reason == "manifest_marks_material_unavailable"


def test_random_noise_material_uses_saved_image_texture(monkeypatch, tmp_path) -> None:
    calls = _install_fake_material_utils(monkeypatch)
    record = {
        "texture_condition": "random_noise",
        "texture_seed": 123,
        "rgb_path": str(tmp_path / "render" / "rgb.png"),
    }

    meta = apply_materials([_FakeObject()], record, {})

    assert meta["material_status"] == "random_noise_override"
    assert meta["random_noise_texture_type"] == "image_texture"
    assert meta["random_texture_path"].endswith("random_texture.png")
    assert calls[0]["texture_type"] == "image_noise"
    assert calls[0]["image_texture_path"] == meta["random_texture_path"]


def test_random_texture_png_writer_outputs_seeded_noise(tmp_path) -> None:
    material_utils = _load_material_utils_module()
    path = tmp_path / "random_texture.png"

    pixels = material_utils._random_rgba_bytes(seed=123, width=16, height=16)
    material_utils._write_rgba_png(path, width=16, height=16, pixels=pixels)

    arr = np.asarray(Image.open(path))
    assert arr.shape == (16, 16, 4)
    assert arr[..., :3].max() > 0
    assert np.all(arr[..., 3] == 255)
    assert len(np.unique(arr[..., :3].reshape(-1, 3), axis=0)) > 1
    assert material_utils._random_rgba_bytes(seed=123, width=16, height=16) == pixels
    assert material_utils._random_rgba_bytes(seed=124, width=16, height=16) != pixels


def test_blender_renderer_outputs_geometry_buffers(tmp_path) -> None:
    blender = shutil.which("blender")
    if blender is None:
        pytest.skip("Blender is not available on PATH")

    repo_root = Path(__file__).resolve().parents[1]
    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)

    out_dir = tmp_path / "render"
    record = {
        "render_id": "toy_box_flat",
        "object_id": "toy_box",
        "source_dataset": "unit",
        "category": "box",
        "split": "train",
        "raw_mesh_path": str(mesh_path),
        "normalized_mesh_path": str(mesh_path),
        "texture_condition": "flat",
        "texture_seed": 0,
        "camera_distance": 2.4,
        "camera_azimuth_deg": 0.0,
        "camera_elevation_deg": 20.0,
        "camera_fov_deg": 50.0,
        "object_scale": 1.0,
        "light_type": "sun",
        "light_azimuth_deg": 45.0,
        "light_elevation_deg": 35.0,
        "light_intensity": 3.0,
        "render_seed": 7,
        "rgb_path": str(out_dir / "rgb.png"),
        "depth_path": str(out_dir / "depth.npy"),
        "normal_path": str(out_dir / "normal_camera.npy"),
        "mask_path": str(out_dir / "mask.npy"),
        "render_status": "pending",
        "qc_error_message": "",
    }
    chunk = tmp_path / "chunk.jsonl"
    chunk.write_text(json.dumps(record) + "\n", encoding="utf-8")
    status = tmp_path / "status.jsonl"
    config = tmp_path / "render.yaml"
    config.write_text(
        "\n".join(
            [
                "render:",
                "  engine: CYCLES",
                "  resolution: [32, 32]",
                "  samples: 1",
                "  use_gpu: false",
                "  transparent_background: false",
                "  file_format: PNG",
                "  camera:",
                "    clip_start: 0.01",
                "    clip_end: 1000.0",
                "  world:",
                "    bg_color: [0.05, 0.05, 0.06, 1.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            blender,
            "--background",
            "--python",
            "scripts/render_blender.py",
            "--",
            "--config",
            str(config),
            "--chunk",
            str(chunk),
            "--status-output",
            str(status),
            "--project-root",
            str(repo_root),
        ],
        cwd=repo_root,
        check=True,
    )

    status_rows = read_jsonl(status)
    assert status_rows[0]["render_status"] == "success"
    assert status_rows[0]["material_status"] == "flat_override"
    assert (out_dir / "rgb.png").is_file()
    assert (out_dir / "render_meta.json").is_file()
    assert np.load(out_dir / "depth.npy").shape == (32, 32)
    assert np.load(out_dir / "normal_camera.npy").shape == (32, 32, 3)
    assert np.load(out_dir / "mask.npy").shape == (32, 32)


def test_blender_material_triplet_preserves_geometry_buffers(tmp_path) -> None:
    blender = shutil.which("blender")
    if blender is None:
        pytest.skip("Blender is not available on PATH")

    repo_root = Path(__file__).resolve().parents[1]
    mesh_path = tmp_path / "chair_0001.obj"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)

    records = []
    for texture in ("photorealistic", "flat", "random_noise"):
        out_dir = tmp_path / texture
        records.append(
            {
                "render_id": f"toy_box_{texture}",
                "object_id": "toy_box",
                "source_dataset": "modelnet40",
                "category": "chair",
                "split": "train",
                "raw_mesh_path": str(tmp_path / "chair_0001.off"),
                "normalized_mesh_path": str(mesh_path),
                "texture_condition": texture,
                "texture_seed": 42,
                "camera_distance": 2.4,
                "camera_azimuth_deg": 0.0,
                "camera_elevation_deg": 20.0,
                "camera_fov_deg": 50.0,
                "object_scale": 1.0,
                "light_type": "sun",
                "light_azimuth_deg": 45.0,
                "light_elevation_deg": 35.0,
                "light_intensity": 3.0,
                "render_seed": 7,
                "rgb_path": str(out_dir / "rgb.png"),
                "depth_path": str(out_dir / "depth.npy"),
                "normal_path": str(out_dir / "normal_camera.npy"),
                "mask_path": str(out_dir / "mask.npy"),
                "render_status": "pending",
                "qc_error_message": "",
            }
        )
    chunk = tmp_path / "chunk.jsonl"
    chunk.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    status = tmp_path / "status.jsonl"
    config = tmp_path / "render.yaml"
    config.write_text(
        "\n".join(
            [
                "render:",
                "  engine: CYCLES",
                "  resolution: [24, 24]",
                "  samples: 1",
                "  use_gpu: false",
                "  transparent_background: false",
                "  file_format: PNG",
                "  camera:",
                "    clip_start: 0.01",
                "    clip_end: 1000.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            blender,
            "--background",
            "--python",
            "scripts/render_blender.py",
            "--",
            "--config",
            str(config),
            "--chunk",
            str(chunk),
            "--status-output",
            str(status),
            "--project-root",
            str(repo_root),
        ],
        cwd=repo_root,
        check=True,
    )

    status_by_texture = {row["texture_condition"]: row for row in read_jsonl(status)}
    assert set(status_by_texture) == {"photorealistic", "flat", "random_noise"}
    assert status_by_texture["photorealistic"]["material_status"] == "fallback"
    assert (
        status_by_texture["photorealistic"]["photorealistic_material_status"]
        == "fallback_missing_original"
    )

    depth_ref = np.load(tmp_path / "flat" / "depth.npy")
    normal_ref = np.load(tmp_path / "flat" / "normal_camera.npy")
    mask_ref = np.load(tmp_path / "flat" / "mask.npy")
    rgbs = {}
    for texture in ("photorealistic", "random_noise"):
        assert np.allclose(
            np.load(tmp_path / texture / "depth.npy"),
            depth_ref,
            equal_nan=True,
        )
        assert np.allclose(
            np.load(tmp_path / texture / "normal_camera.npy"),
            normal_ref,
        )
        assert np.array_equal(np.load(tmp_path / texture / "mask.npy"), mask_ref)
    for texture in ("photorealistic", "flat", "random_noise"):
        rgbs[texture] = np.asarray(Image.open(tmp_path / texture / "rgb.png"))

    assert not np.array_equal(rgbs["photorealistic"], rgbs["flat"])
    assert not np.array_equal(rgbs["flat"], rgbs["random_noise"])

    random_texture = np.asarray(Image.open(tmp_path / "random_noise" / "random_texture.png"))
    assert random_texture[..., :3].max() > 0
    assert len(np.unique(random_texture[..., :3].reshape(-1, 3), axis=0)) > 1

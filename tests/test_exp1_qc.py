from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from exp1.rendering.qc import (
    RenderQCConfig,
    make_contact_sheet,
    validate_render_rows,
    valid_render_rows,
)


def _write_render_outputs(root: Path, *, depth_offset: float = 0.0) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    rgb = np.full((6, 6, 3), 128, dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    depth = np.full((6, 6), np.nan, dtype=np.float32)
    depth[mask] = 2.0 + depth_offset
    normal = np.zeros((6, 6, 3), dtype=np.float32)
    normal[mask, 2] = 1.0

    Image.fromarray(rgb).save(root / "rgb.png")
    np.save(root / "depth.npy", depth)
    np.save(root / "normal_camera.npy", normal)
    np.save(root / "mask.npy", mask)
    return {
        "rgb_path": str(root / "rgb.png"),
        "depth_path": str(root / "depth.npy"),
        "normal_path": str(root / "normal_camera.npy"),
        "mask_path": str(root / "mask.npy"),
    }


def _row(root: Path, render_id: str, *, texture: str = "flat") -> dict:
    return {
        "render_id": render_id,
        "object_id": "obj",
        "split": "train",
        "texture_condition": texture,
        "render_status": "success",
        "texture_control_group_id": "tcg",
        **_write_render_outputs(root / render_id),
    }


def test_validate_render_rows_marks_valid_outputs(tmp_path) -> None:
    rows = [_row(tmp_path, "r_flat")]

    qc = validate_render_rows(
        rows,
        config=RenderQCConfig(min_foreground_fraction=0.1),
    )

    assert bool(qc.loc[0, "qc_pass"]) is True
    assert qc.loc[0, "qc_foreground_fraction"] > 0.1
    assert valid_render_rows(qc.to_dict(orient="records")).shape[0] == 1


def test_validate_render_rows_records_missing_files(tmp_path) -> None:
    rows = [_row(tmp_path, "r_flat")]
    Path(rows[0]["depth_path"]).unlink()

    qc = validate_render_rows(rows)

    assert bool(qc.loc[0, "qc_pass"]) is False
    assert "missing_depth_path" in qc.loc[0, "qc_error_message"]
    assert valid_render_rows(qc.to_dict(orient="records")).empty


def test_texture_control_geometry_mismatch_fails_group(tmp_path) -> None:
    first = _row(tmp_path, "r_flat", texture="flat")
    second = _row(tmp_path, "r_noise", texture="random_noise")
    changed = _write_render_outputs(tmp_path / "r_noise", depth_offset=0.5)
    second.update(changed)

    qc = validate_render_rows([first, second])

    assert set(qc["qc_status"]) == {"failed"}
    assert qc["qc_error_message"].str.contains("texture_control_mismatch").all()


def test_make_contact_sheet_writes_image(tmp_path) -> None:
    rows = [_row(tmp_path, "r_flat"), _row(tmp_path, "r_noise")]
    output = make_contact_sheet(rows, tmp_path / "sheet.png", columns=2)

    assert output.is_file()
    with Image.open(output) as image:
        assert image.size[0] > 0
        assert image.size[1] > 0

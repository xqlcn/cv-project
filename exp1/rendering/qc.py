"""Render quality-control checks for Experiment 1 outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


DEFAULT_CONTROL_TOLERANCES = {
    "depth_atol": 1e-4,
    "normal_atol": 1e-4,
}


@dataclass(frozen=True)
class RenderQCConfig:
    """Thresholds used by render QC."""

    min_foreground_fraction: float = 0.01
    max_foreground_fraction: float = 0.95
    min_depth_valid_fraction: float = 0.9
    normal_norm_tolerance: float = 0.1
    depth_atol: float = 1e-4
    normal_atol: float = 1e-4


class RenderQCError(ValueError):
    """Raised when QC inputs are malformed."""


def _resolve_path(project_root: Optional[Union[str, Path]], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _shape_ok(
    rgb: np.ndarray,
    depth: np.ndarray,
    normal: np.ndarray,
    mask: np.ndarray,
) -> bool:
    height, width = rgb.shape[:2]
    return (
        rgb.ndim == 3
        and rgb.shape[2] in {3, 4}
        and depth.shape == (height, width)
        and normal.shape == (height, width, 3)
        and mask.shape == (height, width)
    )


def qc_render_row(
    row: Mapping[str, Any],
    *,
    project_root: Optional[Union[str, Path]] = None,
    config: RenderQCConfig = RenderQCConfig(),
) -> Dict[str, Any]:
    """Run file and buffer checks for one render row."""
    out = dict(row)
    errors: List[str] = []
    paths = {
        "rgb_path": _resolve_path(project_root, row.get("rgb_path", "")),
        "depth_path": _resolve_path(project_root, row.get("depth_path", "")),
        "normal_path": _resolve_path(project_root, row.get("normal_path", "")),
        "mask_path": _resolve_path(project_root, row.get("mask_path", "")),
    }

    for key, path in paths.items():
        out[f"qc_{key}_exists"] = path.is_file()
        if not path.is_file():
            errors.append(f"missing_{key}:{path}")

    if errors:
        return _finish_qc_row(out, errors, passed=False)

    try:
        rgb = _load_rgb(paths["rgb_path"])
        depth = _load_array(paths["depth_path"])
        normal = _load_array(paths["normal_path"])
        mask = _load_array(paths["mask_path"]).astype(bool)
    except Exception as exc:
        errors.append(f"load_error:{type(exc).__name__}:{exc}")
        return _finish_qc_row(out, errors, passed=False)

    out["qc_rgb_shape"] = tuple(int(v) for v in rgb.shape)
    out["qc_depth_shape"] = tuple(int(v) for v in depth.shape)
    out["qc_normal_shape"] = tuple(int(v) for v in normal.shape)
    out["qc_mask_shape"] = tuple(int(v) for v in mask.shape)

    if not _shape_ok(rgb, depth, normal, mask):
        errors.append("shape_mismatch")
        return _finish_qc_row(out, errors, passed=False)

    foreground_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    foreground_fraction = float(foreground_pixels / max(total_pixels, 1))
    out["qc_foreground_pixel_count"] = foreground_pixels
    out["qc_foreground_fraction"] = foreground_fraction

    if foreground_fraction < config.min_foreground_fraction:
        errors.append("foreground_fraction_too_small")
    if foreground_fraction > config.max_foreground_fraction:
        errors.append("foreground_fraction_too_large")

    foreground_depth = depth[mask]
    finite_depth = np.isfinite(foreground_depth)
    depth_valid_fraction = float(finite_depth.mean()) if foreground_pixels else 0.0
    out["qc_depth_valid_fraction"] = depth_valid_fraction
    out["qc_depth_min"] = (
        float(np.nanmin(foreground_depth)) if finite_depth.any() else np.nan
    )
    out["qc_depth_max"] = (
        float(np.nanmax(foreground_depth)) if finite_depth.any() else np.nan
    )
    if depth_valid_fraction < config.min_depth_valid_fraction:
        errors.append("insufficient_finite_foreground_depth")

    foreground_normal = normal[mask]
    if len(foreground_normal):
        normal_norm = np.linalg.norm(foreground_normal, axis=1)
        finite_norm = normal_norm[np.isfinite(normal_norm)]
    else:
        finite_norm = np.asarray([], dtype=np.float32)

    mean_norm = float(finite_norm.mean()) if len(finite_norm) else np.nan
    out["qc_normal_mean_norm"] = mean_norm
    out["qc_normal_valid_fraction"] = (
        float(len(finite_norm) / len(foreground_normal))
        if len(foreground_normal)
        else 0.0
    )
    if not np.isfinite(mean_norm):
        errors.append("normal_norm_not_finite")
    elif abs(mean_norm - 1.0) > config.normal_norm_tolerance:
        errors.append("normal_norm_out_of_tolerance")

    return _finish_qc_row(out, errors, passed=not errors)


def _finish_qc_row(
    row: Dict[str, Any],
    errors: Sequence[str],
    *,
    passed: bool,
) -> Dict[str, Any]:
    row["qc_pass"] = bool(passed)
    row["qc_status"] = "passed" if passed else "failed"
    row["qc_error_message"] = ";".join(str(error) for error in errors)
    return row


def _finite_close(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
) -> bool:
    finite = np.isfinite(candidate) & np.isfinite(reference)
    nan_match = np.array_equal(np.isnan(candidate), np.isnan(reference))
    if not nan_match:
        return False
    if not finite.any():
        return True
    return bool(np.allclose(candidate[finite], reference[finite], atol=atol))


def _geometry_arrays(
    row: Mapping[str, Any],
    *,
    project_root: Optional[Union[str, Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = _load_array(_resolve_path(project_root, row["depth_path"]))
    normal = _load_array(_resolve_path(project_root, row["normal_path"]))
    mask = _load_array(_resolve_path(project_root, row["mask_path"])).astype(bool)
    return depth, normal, mask


def apply_texture_control_qc(
    rows: Sequence[Mapping[str, Any]],
    *,
    project_root: Optional[Union[str, Path]] = None,
    config: RenderQCConfig = RenderQCConfig(),
    group_column: str = "texture_control_group_id",
) -> List[Dict[str, Any]]:
    """Mark texture-only groups whose depth/normal/mask buffers diverge."""
    out = [dict(row) for row in rows]
    by_group: Dict[str, List[int]] = {}
    for idx, row in enumerate(out):
        group_id = row.get(group_column)
        if group_id is None or str(group_id) == "":
            continue
        by_group.setdefault(str(group_id), []).append(idx)

    for indexes in by_group.values():
        passed_indexes = [
            idx for idx in indexes if bool(out[idx].get("qc_pass", False))
        ]
        if len(passed_indexes) < 2:
            continue
        ref_idx = passed_indexes[0]
        try:
            ref_depth, ref_normal, ref_mask = _geometry_arrays(
                out[ref_idx],
                project_root=project_root,
            )
        except Exception as exc:
            _append_geometry_error(out[ref_idx], f"control_load_error:{exc}")
            continue

        inconsistent_group = False
        for idx in passed_indexes[1:]:
            try:
                depth, normal, mask = _geometry_arrays(
                    out[idx],
                    project_root=project_root,
                )
            except Exception as exc:
                _append_geometry_error(out[idx], f"control_load_error:{exc}")
                inconsistent_group = True
                continue

            ok = (
                np.array_equal(mask, ref_mask)
                and _finite_close(depth, ref_depth, atol=config.depth_atol)
                and np.allclose(normal, ref_normal, atol=config.normal_atol)
            )
            out[idx]["qc_texture_control_depth_atol"] = config.depth_atol
            out[idx]["qc_texture_control_normal_atol"] = config.normal_atol
            out[idx]["qc_texture_control_consistent"] = bool(ok)
            if not ok:
                inconsistent_group = True

        if inconsistent_group:
            for idx in passed_indexes:
                out[idx]["qc_texture_control_consistent"] = False
                if out[idx].get("qc_pass", False):
                    _append_geometry_error(out[idx], "texture_control_mismatch")
        else:
            for idx in passed_indexes:
                out[idx]["qc_texture_control_consistent"] = True

    for row in out:
        row.setdefault("qc_texture_control_consistent", True)
    return out


def _append_geometry_error(row: Dict[str, Any], message: str) -> None:
    existing = str(row.get("qc_error_message", ""))
    row["qc_error_message"] = (
        message if not existing else existing + ";" + message
    )
    row["qc_pass"] = False
    row["qc_status"] = "failed"


def validate_render_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    project_root: Optional[Union[str, Path]] = None,
    config: RenderQCConfig = RenderQCConfig(),
    require_render_success: bool = True,
) -> pd.DataFrame:
    """Return a QC manifest DataFrame for render rows."""
    checked: List[Dict[str, Any]] = []
    for row in rows:
        row_status = str(row.get("render_status", "success"))
        if require_render_success and row_status not in {"success", "passed"}:
            failed = dict(row)
            failed["qc_pass"] = False
            failed["qc_status"] = "failed"
            failed["qc_error_message"] = "render_status_not_success"
            checked.append(failed)
            continue
        checked.append(qc_render_row(row, project_root=project_root, config=config))

    checked = apply_texture_control_qc(
        checked,
        project_root=project_root,
        config=config,
    )
    return pd.DataFrame(checked)


def valid_render_rows(qc_rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Filter QC rows to downstream-eligible successful renders."""
    df = pd.DataFrame(list(qc_rows))
    if df.empty:
        return df
    return df[df["qc_pass"].astype(bool)].reset_index(drop=True)


def make_contact_sheet(
    rows: Iterable[Mapping[str, Any]],
    output_path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
    image_column: str = "rgb_path",
    label_columns: Sequence[str] = ("render_id", "texture_condition", "qc_status"),
    columns: int = 4,
    tile_size: tuple[int, int] = (160, 160),
    max_images: Optional[int] = None,
) -> Path:
    """Create a simple RGB contact sheet from render manifest rows."""
    rows_list = list(rows)
    if max_images is not None:
        rows_list = rows_list[: int(max_images)]
    if not rows_list:
        raise RenderQCError("Cannot create a contact sheet from zero rows")

    columns = max(1, int(columns))
    tile_w, tile_h = [int(v) for v in tile_size]
    label_h = 36
    rows_count = int(np.ceil(len(rows_list) / columns))
    sheet = Image.new(
        "RGB",
        (columns * tile_w, rows_count * (tile_h + label_h)),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)

    for idx, row in enumerate(rows_list):
        image_path = _resolve_path(project_root, row[image_column])
        col = idx % columns
        row_i = idx // columns
        x = col * tile_w
        y = row_i * (tile_h + label_h)

        try:
            with Image.open(image_path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        except Exception:
            thumb = Image.new("RGB", (tile_w, tile_h), color=(60, 60, 60))
        sheet.paste(thumb, (x + (tile_w - thumb.width) // 2, y))

        label = " | ".join(
            str(row.get(column, ""))[:48] for column in label_columns
        )
        draw.text((x + 4, y + tile_h + 4), label[:80], fill=(20, 20, 20))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output

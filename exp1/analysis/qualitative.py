"""Qualitative visual tables for Experiment 1 probe outputs."""

from __future__ import annotations

import math
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

_CACHE_ROOT = Path(tempfile.gettempdir()) / "cv_project_plot_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from exp1.metadata.manifest import load_manifest

TextureSlug = str

TEXTURE_DISPLAY_NAMES = {
    "photorealistic": "Photorealistic",
    "flat": "Flat Textureless",
    "random_noise": "Random Noise",
}

DEFAULT_MODEL_DISPLAY_ORDER = {
    "clip_vit_b16": "CLIP B/16",
    "clip_vit_l14": "CLIP L/14",
    "dinov2_vit_b": "DINOv2 B",
    "dinov2_vit_l": "DINOv2 L",
}


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _resolve_path(project_root: Optional[Union[str, Path]], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


def _read_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq", ".jsonl", ".json"}:
        return load_manifest(path, validate=False)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table extension for qualitative inputs: {path}")


def _fit_image(image: Image.Image, *, size: int = 144) -> np.ndarray:
    image = ImageOps.contain(image.convert("RGB"), (size, size))
    canvas = Image.new("RGB", (size, size), "white")
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image, offset)
    return np.asarray(canvas)


def _placeholder(text: str, *, size: int = 144) -> tuple[np.ndarray, str]:
    image = Image.new("RGB", (size, size), (244, 244, 244))
    return np.asarray(image), text


def _rgb_image(row: Mapping[str, Any], project_root: Path, *, size: int) -> np.ndarray:
    with Image.open(_resolve_path(project_root, row["rgb_path"])) as image:
        return _fit_image(image, size=size)


def _normal_to_rgb(normals: np.ndarray) -> np.ndarray:
    return np.clip((normals.astype(np.float32) * 0.5 + 0.5) * 255.0, 0, 255).astype(
        np.uint8
    )


def _load_mask(mask_path: Union[str, Path], project_root: Path) -> np.ndarray:
    return np.load(_resolve_path(project_root, mask_path), allow_pickle=False).astype(
        bool
    )


def _dense_normal_image(
    manifest_row: Mapping[str, Any],
    project_root: Path,
    *,
    size: int,
) -> np.ndarray:
    """Render the full-resolution surface-normal map (camera frame) as an RGB image."""
    normal_path = manifest_row.get("normal_path")
    mask_path = manifest_row.get("mask_path")
    if not normal_path or not mask_path:
        placeholder, _ = _placeholder("normals\nunavailable", size=size)
        return placeholder
    normals = np.load(_resolve_path(project_root, normal_path), allow_pickle=False)
    mask = _load_mask(mask_path, project_root)
    rgb = _normal_to_rgb(normals)
    rgb[~mask] = 255
    return _fit_image(Image.fromarray(rgb), size=size)


def _dense_depth_image(
    manifest_row: Mapping[str, Any],
    project_root: Path,
    *,
    size: int,
) -> np.ndarray:
    """Render the full-resolution rendered depth map with a perceptual colormap."""
    depth_path = manifest_row.get("depth_path")
    mask_path = manifest_row.get("mask_path")
    if not depth_path or not mask_path:
        placeholder, _ = _placeholder("depth\nunavailable", size=size)
        return placeholder
    depth = np.load(_resolve_path(project_root, depth_path), allow_pickle=False).astype(
        np.float32
    )
    mask = _load_mask(mask_path, project_root)
    valid = mask & np.isfinite(depth)
    if not valid.any():
        placeholder, _ = _placeholder("no foreground", size=size)
        return placeholder
    values = depth[valid]
    lo = float(np.percentile(values, 2.0))
    hi = float(np.percentile(values, 98.0))
    span = max(hi - lo, 1e-6)
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid] = 1.0 - np.clip((depth[valid] - lo) / span, 0.0, 1.0)
    plt = _pyplot()
    cmap = plt.get_cmap("viridis")
    rgba = cmap(normalized)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 255
    return _fit_image(Image.fromarray(rgb), size=size)


def _pair_count(row: pd.Series) -> int:
    if "relative_depth_pair_count" in row and pd.notna(row["relative_depth_pair_count"]):
        return int(row["relative_depth_pair_count"])
    return len(
        [
            column
            for column in row.index
            if str(column).startswith("pair_") and str(column).endswith("_label")
        ]
    )


def _relative_depth_scores(
    label_row: pd.Series,
    *,
    pred_row: Optional[pd.Series] = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows = int(label_row.get("relative_depth_grid_rows", 3))
    cols = int(label_row.get("relative_depth_grid_cols", 3))
    scores = np.zeros(rows * cols, dtype=np.float32)
    counts = np.zeros(rows * cols, dtype=np.float32)
    for idx in range(_pair_count(label_row)):
        valid_key = f"pair_{idx}_valid"
        if valid_key in label_row and not bool(label_row[valid_key]):
            continue
        a = int(label_row.get(f"pair_{idx}_region_a", 0))
        b = int(label_row.get(f"pair_{idx}_region_b", 0))
        if pred_row is None:
            value = float(label_row.get(f"pair_{idx}_label", np.nan))
        else:
            value = float(pred_row.get(f"pair_{idx}_prob", np.nan))
        if not np.isfinite(value):
            continue
        scores[a] += value
        scores[b] += 1.0 - value
        counts[a] += 1.0
        counts[b] += 1.0
    grid = np.divide(
        scores,
        counts,
        out=np.full_like(scores, np.nan, dtype=np.float32),
        where=counts > 0,
    ).reshape(rows, cols)
    return grid, counts.reshape(rows, cols) > 0


def _relative_depth_pair_overlay(
    label_row: pd.Series,
    pred_row: pd.Series,
    *,
    rgb_image: np.ndarray,
    size: int,
) -> tuple[np.ndarray, Optional[str]]:
    """Overlay the 3x3 region grid with valid pair predictions on the rendered image."""
    rows = int(label_row.get("relative_depth_grid_rows", 3))
    cols = int(label_row.get("relative_depth_grid_cols", 3))
    plt = _pyplot()
    fig = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(rgb_image)
    ax.set_xticks([])
    ax.set_yticks([])
    h, w = rgb_image.shape[:2]
    cell_h = h / rows
    cell_w = w / cols
    for r in range(1, rows):
        ax.axhline(r * cell_h, color=(1.0, 1.0, 1.0, 0.6), linewidth=0.6)
    for c in range(1, cols):
        ax.axvline(c * cell_w, color=(1.0, 1.0, 1.0, 0.6), linewidth=0.6)

    def cell_center(idx: int) -> tuple[float, float]:
        rr, cc = divmod(idx, cols)
        return (cc + 0.5) * cell_w, (rr + 0.5) * cell_h

    correct = 0
    total = 0
    for idx in range(_pair_count(label_row)):
        if f"pair_{idx}_valid" not in label_row:
            continue
        if not bool(label_row[f"pair_{idx}_valid"]):
            continue
        if not bool(pred_row.get(f"pair_{idx}_valid", True)):
            continue
        a = int(label_row.get(f"pair_{idx}_region_a", 0))
        b = int(label_row.get(f"pair_{idx}_region_b", 0))
        label = float(label_row.get(f"pair_{idx}_label", np.nan))
        prob = float(pred_row.get(f"pair_{idx}_prob", np.nan))
        if not np.isfinite(label) or not np.isfinite(prob):
            continue
        total += 1
        pred_a_closer = prob >= 0.5
        gt_a_closer = label >= 0.5
        is_correct = pred_a_closer == gt_a_closer
        if is_correct:
            correct += 1
        if pred_a_closer:
            src, dst = cell_center(a), cell_center(b)
        else:
            src, dst = cell_center(b), cell_center(a)
        color = (0.16, 0.74, 0.34) if is_correct else (0.86, 0.20, 0.27)
        ax.annotate(
            "",
            xy=dst,
            xytext=src,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.6, alpha=0.95, shrinkA=8, shrinkB=8
            ),
        )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[
        ..., :3
    ].copy()
    plt.close(fig)
    caption = f"{correct}/{total} pairs" if total else None
    return _fit_image(Image.fromarray(image), size=size), caption




def _prediction_path(
    probe_root: Path,
    *,
    model_name: str,
    layer_name: str,
    task: str,
    texture: str,
) -> Optional[Path]:
    candidates = [
        probe_root
        / model_name
        / layer_name
        / task
        / f"within_{texture}"
        / "predictions.csv",
        probe_root
        / model_name
        / layer_name
        / task
        / f"texture_{texture}"
        / "predictions.csv",
        probe_root / model_name / layer_name / task / "predictions.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_prediction_rows(
    probe_root: Path,
    *,
    model_names: Sequence[str],
    layer_name: str,
    task: str,
    textures: Sequence[str],
) -> dict[tuple[str, str], pd.DataFrame]:
    predictions = {}
    for model_name in model_names:
        for texture in textures:
            path = _prediction_path(
                probe_root,
                model_name=model_name,
                layer_name=layer_name,
                task=task,
                texture=texture,
            )
            if path is None:
                continue
            predictions[(model_name, texture)] = pd.read_csv(path)
    return predictions


def _control_group_column(manifest: pd.DataFrame) -> Optional[str]:
    for column in (
        "texture_control_group_id",
        "control_group_id",
        "render_setting_id",
    ):
        if column in manifest.columns:
            return column
    return None


def _fallback_group_columns(manifest: pd.DataFrame) -> list[str]:
    candidates = [
        "source_dataset",
        "object_id",
        "category",
        "split",
        "camera_distance",
        "camera_azimuth_deg",
        "camera_elevation_deg",
        "camera_fov_deg",
        "light_type",
        "light_azimuth_deg",
        "light_elevation_deg",
        "light_intensity",
        "object_scale",
    ]
    return [column for column in candidates if column in manifest.columns]


def _select_texture_triplet(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    textures: Sequence[str],
    preferred_split: str,
) -> dict[str, pd.Series]:
    if "label_valid" in labels.columns:
        label_mask = labels["label_valid"].astype(bool)
    else:
        label_mask = pd.Series(True, index=labels.index)
    valid_label_ids = set(labels.loc[label_mask, "render_id"].astype(str))
    rows = manifest[manifest["render_id"].astype(str).isin(valid_label_ids)].copy()
    rows = rows[rows["texture_condition"].astype(str).isin(set(textures))]
    split_order = [preferred_split] + [
        split
        for split in ("test", "val", "train")
        if split != preferred_split and split in set(rows["split"].astype(str))
    ]
    group_column = _control_group_column(rows)
    if group_column is None:
        group_cols = _fallback_group_columns(rows)
        rows["_qualitative_group"] = rows[group_cols].astype(str).agg("|".join, axis=1)
        group_column = "_qualitative_group"

    for split in split_order:
        split_rows = rows[rows["split"].astype(str) == split]
        for _, group in split_rows.groupby(group_column, sort=True):
            by_texture = {}
            for texture in textures:
                texture_rows = group[group["texture_condition"].astype(str) == texture]
                if texture_rows.empty:
                    break
                by_texture[texture] = texture_rows.sort_values("render_id").iloc[0]
            if len(by_texture) == len(textures):
                return by_texture
    raise ValueError("Could not find a label-valid texture triplet for qualitative table")


def _labels_by_render_id(labels: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        str(row["render_id"]): row
        for _, row in labels.drop_duplicates("render_id").iterrows()
    }


def _fmt_float(value: Any, *, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _angle_from_sincos(sin_value: Any, cos_value: Any) -> float:
    return math.degrees(math.atan2(float(sin_value), float(cos_value)))


def _task_ground_truth_text(task: str, label_row: pd.Series) -> str:
    if task == "viewpoint":
        return "\n".join(
            [
                "GT",
                f"az {_fmt_float(label_row.get('camera_azimuth_deg'), digits=1)} deg",
                f"el {_fmt_float(label_row.get('camera_elevation_deg'), digits=1)} deg",
            ]
        )
    if task == "lighting_direction":
        return "\n".join(
            [
                "GT light dir",
                f"az {_fmt_float(label_row.get('light_azimuth_deg'), digits=1)} deg",
                f"el {_fmt_float(label_row.get('light_elevation_deg'), digits=1)} deg",
                f"xyz ({_fmt_float(label_row.get('light_dir_x'))}, "
                f"{_fmt_float(label_row.get('light_dir_y'))}, "
                f"{_fmt_float(label_row.get('light_dir_z'))})",
            ]
        )
    if task == "lighting_intensity":
        return "\n".join(
            [
                "GT intensity",
                _fmt_float(label_row.get("light_intensity")),
                f"log {_fmt_float(label_row.get('log_light_intensity'))}",
            ]
        )
    if task == "apparent_scale":
        return "\n".join(
            [
                "GT scale",
                _fmt_float(label_row.get("object_scale")),
                f"area {_fmt_float(label_row.get('foreground_area_fraction'), digits=3)}",
            ]
        )
    return "unsupported"


def _task_prediction_text(task: str, pred_row: pd.Series) -> str:
    if task == "viewpoint":
        pred_azimuth = _angle_from_sincos(
            pred_row.get("pred_azimuth_sin"), pred_row.get("pred_azimuth_cos")
        )
        target_azimuth = _angle_from_sincos(
            pred_row.get("target_azimuth_sin"), pred_row.get("target_azimuth_cos")
        )
        pred_elevation = _angle_from_sincos(
            pred_row.get("pred_elevation_sin"), pred_row.get("pred_elevation_cos")
        )
        target_elevation = _angle_from_sincos(
            pred_row.get("target_elevation_sin"), pred_row.get("target_elevation_cos")
        )
        return "\n".join(
            [
                f"pred az {pred_azimuth:.1f} deg",
                f"GT az {target_azimuth:.1f} deg",
                f"pred el {pred_elevation:.1f} deg",
                f"GT el {target_elevation:.1f} deg",
                f"err {_fmt_float(pred_row.get('viewpoint_angular_error_deg'), digits=1)} deg",
            ]
        )
    if task == "lighting_direction":
        return "\n".join(
            [
                "pred light dir",
                f"xyz ({_fmt_float(pred_row.get('pred_light_dir_x'))}, "
                f"{_fmt_float(pred_row.get('pred_light_dir_y'))}, "
                f"{_fmt_float(pred_row.get('pred_light_dir_z'))})",
                "GT",
                f"xyz ({_fmt_float(pred_row.get('target_light_dir_x'))}, "
                f"{_fmt_float(pred_row.get('target_light_dir_y'))}, "
                f"{_fmt_float(pred_row.get('target_light_dir_z'))})",
                f"err {_fmt_float(pred_row.get('angular_error_deg'), digits=1)} deg",
            ]
        )
    if task == "lighting_intensity":
        pred_log = pred_row.get("pred_log_light_intensity")
        target_log = pred_row.get("target_log_light_intensity")
        try:
            pred_intensity = math.exp(float(pred_log))
            target_intensity = math.exp(float(target_log))
        except (TypeError, ValueError, OverflowError):
            pred_intensity = np.nan
            target_intensity = np.nan
        return "\n".join(
            [
                f"pred {_fmt_float(pred_intensity)}",
                f"GT {_fmt_float(target_intensity)}",
                f"log err {_fmt_float(pred_row.get('abs_error_log_light_intensity'))}",
            ]
        )
    if task == "apparent_scale":
        return "\n".join(
            [
                f"pred scale {_fmt_float(pred_row.get('pred_object_scale'))}",
                f"GT scale {_fmt_float(pred_row.get('target_object_scale'))}",
                f"pred area {_fmt_float(pred_row.get('pred_foreground_area_fraction'), digits=3)}",
                f"GT area {_fmt_float(pred_row.get('target_foreground_area_fraction'), digits=3)}",
                f"MAE {_fmt_float(pred_row.get('row_mae'))}",
            ]
        )
    return "unsupported"


def make_qualitative_probe_table(
    *,
    task: str,
    manifest_path: Union[str, Path],
    label_path: Union[str, Path],
    probe_root: Union[str, Path],
    output_path: Union[str, Path],
    project_root: Union[str, Path],
    model_display_order: Optional[Mapping[str, str]] = None,
    layer_name: str = "final",
    textures: Sequence[str] = ("photorealistic", "flat", "random_noise"),
    preferred_split: str = "test",
    image_size: int = 200,
) -> Path:
    """Create one visual qualitative table for a configured probe task."""
    project_root = Path(project_root)
    probe_root = Path(probe_root)
    output_path = Path(output_path)
    model_display_order = dict(model_display_order or DEFAULT_MODEL_DISPLAY_ORDER)
    model_names = list(model_display_order.keys())

    manifest = load_manifest(manifest_path, validate=False)
    labels = _read_table(label_path)
    triplet = _select_texture_triplet(
        manifest,
        labels,
        textures=textures,
        preferred_split=preferred_split,
    )
    labels_lookup = _labels_by_render_id(labels)
    predictions = _load_prediction_rows(
        probe_root,
        model_names=model_names,
        layer_name=layer_name,
        task=task,
        textures=textures,
    )

    columns = ["Texture Type", "Input Image", "Ground Truth"]
    columns.extend(model_display_order.values())
    plt = _pyplot()
    fig, axes = plt.subplots(
        len(textures),
        len(columns),
        figsize=(1.85 * len(columns), 2.15 * len(textures)),
        squeeze=False,
    )

    for row_idx, texture in enumerate(textures):
        manifest_row = triplet[texture]
        render_id = str(manifest_row["render_id"])
        label_row = labels_lookup[render_id]
        rgb = _rgb_image(manifest_row, project_root, size=image_size)
        row_images: list[tuple[Optional[np.ndarray], Optional[str]]] = [
            (None, TEXTURE_DISPLAY_NAMES.get(texture, texture)),
            (rgb, None),
        ]
        if task == "relative_depth_regions":
            row_images.append(
                (
                    _dense_depth_image(manifest_row, project_root, size=image_size),
                    "dense depth (closer=yellow)",
                )
            )
        else:
            row_images.append((None, _task_ground_truth_text(task, label_row)))

        for model_name in model_names:
            pred_df = predictions.get((model_name, texture))
            if pred_df is None:
                image, text = _placeholder("not run", size=image_size)
                row_images.append((image, text))
                continue
            candidates = pred_df[
                (pred_df["render_id"].astype(str) == render_id)
                & (pred_df["split"].astype(str) == str(manifest_row["split"]))
            ]
            if candidates.empty:
                image, text = _placeholder("no row", size=image_size)
                row_images.append((image, text))
                continue
            pred_row = candidates.iloc[0]
            if task == "relative_depth_regions":
                image, caption = _relative_depth_pair_overlay(
                    label_row,
                    pred_row,
                    rgb_image=rgb,
                    size=image_size,
                )
                row_images.append((image, caption))
            else:
                row_images.append((None, _task_prediction_text(task, pred_row)))

        for col_idx, (image, text) in enumerate(row_images):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if image is not None:
                ax.imshow(image)
            if text:
                if image is None:
                    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
                else:
                    ax.text(
                        0.5,
                        -0.06,
                        text,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=7,
                        color="#444",
                    )
            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=9)

    title = task.replace("_", " ")
    if task == "relative_depth_regions":
        subtitle = (
            "3x3 region pair ordering; arrows point from closer to farther region. "
            "Green = probe matches ground truth, red = wrong, yellow = ground truth."
        )
    else:
        subtitle = ""
    fig.suptitle(
        f"{title} qualitative examples ({layer_name})\n{subtitle}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _depth_prediction_path(
    probe_root: Path,
    *,
    model_name: str,
    layer_name: str,
    texture: str,
    split: str,
) -> Optional[Path]:
    candidates = [
        probe_root
        / model_name
        / layer_name
        / "dense_depth_patches"
        / f"within_{texture}"
        / f"predictions_{split}.npz",
        probe_root
        / model_name
        / layer_name
        / "dense_depth_patches"
        / f"texture_{texture}"
        / f"predictions_{split}.npz",
        probe_root
        / model_name
        / layer_name
        / "dense_depth_patches"
        / "all_textures"
        / f"predictions_{split}.npz",
        probe_root
        / model_name
        / layer_name
        / "dense_relative_depth"
        / f"within_{texture}"
        / f"predictions_{split}.npz",
        probe_root
        / model_name
        / layer_name
        / "dense_relative_depth"
        / f"texture_{texture}"
        / f"predictions_{split}.npz",
        probe_root
        / model_name
        / layer_name
        / "dense_relative_depth"
        / "all_textures"
        / f"predictions_{split}.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _normal_prediction_path(
    probe_root: Path,
    *,
    model_name: str,
    layer_name: str,
    texture: str,
    split: str,
) -> Optional[Path]:
    candidates = [
        probe_root
        / model_name
        / layer_name
        / "dense_surface_normal_patches"
        / f"within_{texture}"
        / f"predictions_{split}.npz",
        probe_root
        / model_name
        / layer_name
        / "dense_surface_normal_patches"
        / f"texture_{texture}"
        / f"predictions_{split}.npz",
        probe_root
        / model_name
        / layer_name
        / "dense_surface_normal_patches"
        / "all_textures"
        / f"predictions_{split}.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _normalize_depth(
    values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Normalize depth to [0,1] with percentile clipping over valid pixels."""
    valid = mask & np.isfinite(values)
    if not valid.any():
        return np.zeros_like(values, dtype=np.float32), 0.0, 1.0
    sample = values[valid]
    lo = float(np.percentile(sample, 2.0))
    hi = float(np.percentile(sample, 98.0))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    out = np.zeros_like(values, dtype=np.float32)
    out[valid] = np.clip((sample - lo) / (hi - lo), 0.0, 1.0)
    return out, lo, hi


def _depth_to_rgb(values: np.ndarray, mask: np.ndarray, *, invert: bool) -> np.ndarray:
    plt = _pyplot()
    cmap = plt.get_cmap("viridis")
    normalized, _, _ = _normalize_depth(values, mask)
    if invert:
        normalized = 1.0 - normalized
        normalized[~mask] = 0.0
    rgba = cmap(normalized)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~mask] = 255
    return rgb


def _dense_depth_gt_image(
    manifest_row: Mapping[str, Any],
    project_root: Path,
    *,
    size: int,
) -> np.ndarray:
    """Full-resolution rendered depth map, colorized with yellow=closer."""
    depth_path = manifest_row.get("depth_path")
    mask_path = manifest_row.get("mask_path")
    if not depth_path or not mask_path:
        placeholder, _ = _placeholder("depth\nunavailable", size=size)
        return placeholder
    depth = np.load(_resolve_path(project_root, depth_path), allow_pickle=False).astype(
        np.float32
    )
    mask = _load_mask(mask_path, project_root)
    valid = mask & np.isfinite(depth)
    rgb = _depth_to_rgb(depth, valid, invert=True)
    return _fit_image(Image.fromarray(rgb), size=size)


def _dense_object_mask(
    manifest_row: Mapping[str, Any],
    project_root: Path,
    *,
    target_resolution: int,
) -> Optional[np.ndarray]:
    """Return full-resolution foreground mask oriented to match the RGB image."""
    mask_path = manifest_row.get("mask_path")
    if not mask_path:
        return None
    mask = _load_mask(mask_path, project_root)
    if mask.shape != (target_resolution, target_resolution):
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (target_resolution, target_resolution),
            Image.NEAREST,
        )
        mask = np.asarray(mask_image, dtype=np.uint8) > 127
    return mask


def _dense_prediction_image(
    pred_grid: np.ndarray,
    valid_grid: np.ndarray,
    *,
    image_size: int,
    target_resolution: int = 224,
    object_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bilinearly upsample a (P,P) predicted patch grid to image_size and colorize."""
    if pred_grid.size == 0:
        placeholder, _ = _placeholder("no\nprediction", size=image_size)
        return placeholder
    pred = pred_grid.astype(np.float32)
    valid_bool = valid_grid.astype(bool)
    pil = Image.fromarray(pred).resize(
        (target_resolution, target_resolution), Image.BILINEAR
    )
    upsampled = np.asarray(pil, dtype=np.float32)
    valid_pil = Image.fromarray(valid_bool.astype(np.uint8) * 255).resize(
        (target_resolution, target_resolution), Image.NEAREST
    )
    upsampled_mask = np.asarray(valid_pil, dtype=np.uint8) > 127
    if object_mask is not None:
        upsampled_mask = upsampled_mask & object_mask.astype(bool)
    rgb = _depth_to_rgb(upsampled, upsampled_mask, invert=True)
    return _fit_image(Image.fromarray(rgb), size=image_size)


def _dense_normal_prediction_image(
    pred_grid: np.ndarray,
    valid_grid: np.ndarray,
    *,
    image_size: int,
    target_resolution: int = 224,
    object_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bilinearly upsample a (P,P,3) predicted normal grid and colorize."""
    if pred_grid.size == 0:
        placeholder, _ = _placeholder("no\nprediction", size=image_size)
        return placeholder
    pred = np.asarray(pred_grid, dtype=np.float32)
    channels = []
    for idx in range(3):
        channel = Image.fromarray(pred[..., idx]).resize(
            (target_resolution, target_resolution),
            Image.BILINEAR,
        )
        channels.append(np.asarray(channel, dtype=np.float32))
    upsampled = np.stack(channels, axis=-1)
    norm = np.linalg.norm(upsampled, axis=-1, keepdims=True)
    upsampled = np.divide(
        upsampled,
        np.clip(norm, 1e-8, None),
        out=np.zeros_like(upsampled),
        where=np.isfinite(norm),
    )
    valid_pil = Image.fromarray(valid_grid.astype(np.uint8) * 255).resize(
        (target_resolution, target_resolution),
        Image.NEAREST,
    )
    upsampled_mask = np.asarray(valid_pil, dtype=np.uint8) > 127
    if object_mask is not None:
        upsampled_mask = upsampled_mask & object_mask.astype(bool)
    rgb = _normal_to_rgb(upsampled)
    rgb[~upsampled_mask] = 255
    return _fit_image(Image.fromarray(rgb), size=image_size)


def _align_prediction_to_target(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Solve (scale, shift) so scale*pred + shift ~ target on valid pixels."""
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    if not valid.any():
        return pred.astype(np.float32)
    p = pred[valid].astype(np.float64)
    t = target[valid].astype(np.float64)
    if p.size < 2:
        return pred.astype(np.float32)
    try:
        design = np.stack([p, np.ones_like(p)], axis=1)
        coef, *_ = np.linalg.lstsq(design, t, rcond=None)
        scale, shift = float(coef[0]), float(coef[1])
    except np.linalg.LinAlgError:
        return pred.astype(np.float32)
    return (pred.astype(np.float32) * scale + shift).astype(np.float32)


def _load_dense_predictions_for_models(
    probe_root: Path,
    *,
    model_names: Sequence[str],
    layer_name: str,
    textures: Sequence[str],
    splits: Sequence[str] = ("test", "val", "train"),
) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    """Load dense-depth predictions keyed by (model, texture, split)."""
    payloads: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for model_name in model_names:
        for texture in textures:
            for split in splits:
                path = _depth_prediction_path(
                    probe_root,
                    model_name=model_name,
                    layer_name=layer_name,
                    texture=texture,
                    split=split,
                )
                if path is None:
                    continue
                with np.load(path, allow_pickle=False) as data:
                    payloads[(model_name, texture, split)] = {
                        "render_ids": np.asarray(data["render_ids"], dtype=str),
                        "predictions": np.asarray(data["predictions"], dtype=np.float32),
                        "targets": np.asarray(data["targets"], dtype=np.float32),
                        "valid": np.asarray(data["valid"], dtype=bool),
                    }
    return payloads


def _load_dense_normal_predictions_for_models(
    probe_root: Path,
    *,
    model_names: Sequence[str],
    layer_name: str,
    textures: Sequence[str],
    splits: Sequence[str] = ("test", "val", "train"),
) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    """Load dense-normal predictions keyed by (model, texture, split)."""
    payloads: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for model_name in model_names:
        for texture in textures:
            for split in splits:
                path = _normal_prediction_path(
                    probe_root,
                    model_name=model_name,
                    layer_name=layer_name,
                    texture=texture,
                    split=split,
                )
                if path is None:
                    continue
                with np.load(path, allow_pickle=False) as data:
                    payloads[(model_name, texture, split)] = {
                        "render_ids": np.asarray(data["render_ids"], dtype=str),
                        "predictions": np.asarray(data["predictions"], dtype=np.float32),
                        "targets": np.asarray(data["targets"], dtype=np.float32),
                        "valid": np.asarray(data["valid"], dtype=bool),
                    }
    return payloads


def make_dense_surface_normal_qualitative_table(
    *,
    manifest_path: Union[str, Path],
    probe_root: Union[str, Path],
    output_path: Union[str, Path],
    project_root: Union[str, Path],
    model_display_order: Optional[Mapping[str, str]] = None,
    layer_name: str = "final",
    textures: Sequence[str] = ("photorealistic", "flat", "random_noise"),
    preferred_split: str = "test",
    image_size: int = 200,
) -> Optional[Path]:
    """Create a dense surface-normal qualitative table."""
    project_root = Path(project_root)
    probe_root = Path(probe_root)
    output_path = Path(output_path)
    model_display_order = dict(model_display_order or DEFAULT_MODEL_DISPLAY_ORDER)
    model_names = list(model_display_order.keys())

    manifest = load_manifest(manifest_path, validate=False)
    predictions_by_key = _load_dense_normal_predictions_for_models(
        probe_root,
        model_names=model_names,
        layer_name=layer_name,
        textures=textures,
    )
    if not predictions_by_key:
        return None

    available_render_ids: set[str] = set()
    for payload in predictions_by_key.values():
        available_render_ids.update(payload["render_ids"].tolist())
    rows = manifest[manifest["render_id"].astype(str).isin(available_render_ids)].copy()
    rows = rows[rows["texture_condition"].astype(str).isin(set(textures))]
    split_order = [preferred_split] + [
        split
        for split in ("test", "val", "train")
        if split != preferred_split and split in set(rows["split"].astype(str))
    ]
    group_column = _control_group_column(rows)
    if group_column is None:
        group_cols = _fallback_group_columns(rows)
        rows["_qualitative_group"] = rows[group_cols].astype(str).agg("|".join, axis=1)
        group_column = "_qualitative_group"

    triplet: Optional[dict[str, pd.Series]] = None
    chosen_split: Optional[str] = None
    for split in split_order:
        split_rows = rows[rows["split"].astype(str) == split]
        for _, group in split_rows.groupby(group_column, sort=True):
            by_texture: dict[str, pd.Series] = {}
            for texture in textures:
                texture_rows = group[group["texture_condition"].astype(str) == texture]
                if texture_rows.empty:
                    break
                by_texture[texture] = texture_rows.sort_values("render_id").iloc[0]
            if len(by_texture) == len(textures):
                triplet = by_texture
                chosen_split = split
                break
        if triplet is not None:
            break
    if triplet is None or chosen_split is None:
        return None

    columns = ["Texture Type", "Input Image", "Ground Truth"]
    columns.extend(model_display_order.values())
    plt = _pyplot()
    fig, axes = plt.subplots(
        len(textures),
        len(columns),
        figsize=(2.1 * len(columns), 2.45 * len(textures)),
        squeeze=False,
    )
    for row_idx, texture in enumerate(textures):
        manifest_row = triplet[texture]
        render_id = str(manifest_row["render_id"])
        object_mask = _dense_object_mask(
            manifest_row,
            project_root,
            target_resolution=224,
        )
        row_cells: list[tuple[Optional[np.ndarray], Optional[str]]] = [
            (None, TEXTURE_DISPLAY_NAMES.get(texture, texture)),
            (_rgb_image(manifest_row, project_root, size=image_size), None),
            (_dense_normal_image(manifest_row, project_root, size=image_size), "normal map"),
        ]
        for model_name in model_names:
            payload = predictions_by_key.get((model_name, texture, chosen_split))
            if payload is None:
                for split in ("test", "val", "train"):
                    payload = predictions_by_key.get((model_name, texture, split))
                    if payload is not None:
                        break
            if payload is None:
                row_cells.append(_placeholder("not run", size=image_size))
                continue
            matches = np.where(payload["render_ids"] == render_id)[0]
            if len(matches) == 0:
                row_cells.append(_placeholder("no row", size=image_size))
                continue
            idx = int(matches[0])
            pred_grid = payload["predictions"][idx]
            valid_grid = payload["valid"][idx]
            target_grid = payload["targets"][idx]
            image = _dense_normal_prediction_image(
                pred_grid,
                valid_grid,
                image_size=image_size,
                object_mask=object_mask,
            )
            valid_flat = valid_grid & np.isfinite(target_grid).all(axis=-1)
            caption: Optional[str] = None
            if valid_flat.any():
                pred_unit = pred_grid / np.clip(
                    np.linalg.norm(pred_grid, axis=-1, keepdims=True),
                    1e-8,
                    None,
                )
                cos = np.clip((pred_unit * target_grid).sum(axis=-1), -1.0, 1.0)
                err = np.degrees(np.arccos(cos))[valid_flat]
                caption = f"err {float(err.mean()):.1f} deg"
            row_cells.append((image, caption))

        for col_idx, (image, text) in enumerate(row_cells):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if image is not None:
                ax.imshow(image)
            if text:
                if image is None:
                    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
                else:
                    ax.text(
                        0.5,
                        -0.10,
                        text,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=7,
                        color="#444",
                    )
            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=9)

    subtitle = (
        "Patch-normal probes are bilinearly upsampled and masked to the rendered "
        "foreground. Colors encode camera-frame normals as xyz -> RGB."
    )
    fig.suptitle(
        f"dense surface-normal qualitative examples ({layer_name})\n"
        + textwrap.fill(subtitle, width=110),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_dense_depth_qualitative_table(
    *,
    manifest_path: Union[str, Path],
    probe_root: Union[str, Path],
    output_path: Union[str, Path],
    project_root: Union[str, Path],
    model_display_order: Optional[Mapping[str, str]] = None,
    layer_name: str = "final",
    textures: Sequence[str] = ("photorealistic", "flat", "random_noise"),
    preferred_split: str = "test",
    image_size: int = 200,
) -> Optional[Path]:
    """Create a dense-depth qualitative table at full image resolution."""
    project_root = Path(project_root)
    probe_root = Path(probe_root)
    output_path = Path(output_path)
    model_display_order = dict(model_display_order or DEFAULT_MODEL_DISPLAY_ORDER)
    model_names = list(model_display_order.keys())

    manifest = load_manifest(manifest_path, validate=False)
    predictions_by_key = _load_dense_predictions_for_models(
        probe_root,
        model_names=model_names,
        layer_name=layer_name,
        textures=textures,
    )
    if not predictions_by_key:
        return None

    available_render_ids: set[str] = set()
    for payload in predictions_by_key.values():
        available_render_ids.update(payload["render_ids"].tolist())
    if not available_render_ids:
        return None

    rows = manifest[manifest["render_id"].astype(str).isin(available_render_ids)].copy()
    rows = rows[rows["texture_condition"].astype(str).isin(set(textures))]
    split_order = [preferred_split] + [
        split
        for split in ("test", "val", "train")
        if split != preferred_split and split in set(rows["split"].astype(str))
    ]
    group_column = _control_group_column(rows)
    if group_column is None:
        group_cols = _fallback_group_columns(rows)
        rows["_qualitative_group"] = rows[group_cols].astype(str).agg("|".join, axis=1)
        group_column = "_qualitative_group"

    triplet: Optional[dict[str, pd.Series]] = None
    chosen_split: Optional[str] = None
    for split in split_order:
        split_rows = rows[rows["split"].astype(str) == split]
        for _, group in split_rows.groupby(group_column, sort=True):
            by_texture: dict[str, pd.Series] = {}
            for texture in textures:
                texture_rows = group[group["texture_condition"].astype(str) == texture]
                if texture_rows.empty:
                    break
                by_texture[texture] = texture_rows.sort_values("render_id").iloc[0]
            if len(by_texture) == len(textures):
                triplet = by_texture
                chosen_split = split
                break
        if triplet is not None:
            break
    if triplet is None or chosen_split is None:
        return None

    columns = ["Texture Type", "Input Image", "Ground Truth"]
    columns.extend(model_display_order.values())
    plt = _pyplot()
    fig, axes = plt.subplots(
        len(textures),
        len(columns),
        figsize=(2.1 * len(columns), 2.45 * len(textures)),
        squeeze=False,
    )
    for row_idx, texture in enumerate(textures):
        manifest_row = triplet[texture]
        render_id = str(manifest_row["render_id"])
        rgb = _rgb_image(manifest_row, project_root, size=image_size)
        object_mask = _dense_object_mask(
            manifest_row,
            project_root,
            target_resolution=224,
        )
        row_cells: list[tuple[Optional[np.ndarray], Optional[str]]] = [
            (None, TEXTURE_DISPLAY_NAMES.get(texture, texture)),
            (rgb, None),
            (
                _dense_depth_gt_image(manifest_row, project_root, size=image_size),
                "rendered depth",
            ),
        ]
        for model_name in model_names:
            payload = predictions_by_key.get((model_name, texture, chosen_split))
            if payload is None:
                # Try other splits if this model is missing the preferred one.
                for split in ("test", "val", "train"):
                    payload = predictions_by_key.get((model_name, texture, split))
                    if payload is not None:
                        break
            if payload is None:
                placeholder, text = _placeholder("not run", size=image_size)
                row_cells.append((placeholder, text))
                continue
            ids = payload["render_ids"]
            matches = np.where(ids == render_id)[0]
            if len(matches) == 0:
                placeholder, text = _placeholder("no row", size=image_size)
                row_cells.append((placeholder, text))
                continue
            idx = int(matches[0])
            pred_grid = payload["predictions"][idx]
            valid_grid = payload["valid"][idx]
            target_grid = payload["targets"][idx]
            aligned = _align_prediction_to_target(pred_grid, target_grid, valid_grid)
            image = _dense_prediction_image(
                aligned,
                valid_grid,
                image_size=image_size,
                object_mask=object_mask,
            )
            # Pearson r over valid patches gives a quick read of fit quality.
            valid_flat = valid_grid & np.isfinite(target_grid) & np.isfinite(pred_grid)
            if valid_flat.sum() > 1:
                p = pred_grid[valid_flat]
                t = target_grid[valid_flat]
                if p.std() > 0 and t.std() > 0:
                    r = float(np.corrcoef(p, t)[0, 1])
                    caption: Optional[str] = f"r = {r:+.2f}"
                else:
                    caption = None
            else:
                caption = None
            row_cells.append((image, caption))

        for col_idx, (image, text) in enumerate(row_cells):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if image is not None:
                ax.imshow(image)
            if text:
                if image is None:
                    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
                else:
                    ax.text(
                        0.5,
                        -0.10,
                        text,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=7,
                        color="#444",
                    )
            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=9)

    subtitle = (
        "Patch-depth probes are bilinearly upsampled, masked to the rendered "
        "foreground, and scale/shift aligned to ground truth for visualization. "
        "r = Pearson correlation over valid patches."
    )
    fig.suptitle(
        f"dense relative depth qualitative examples ({layer_name})\n"
        + textwrap.fill(subtitle, width=110),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_qualitative_probe_tables(
    *,
    tasks: Sequence[str],
    task_label_paths: Mapping[str, Union[str, Path]],
    manifest_path: Union[str, Path],
    probe_root: Union[str, Path],
    output_dir: Union[str, Path],
    project_root: Union[str, Path],
    model_display_order: Optional[Mapping[str, str]] = None,
    layer_name: str = "final",
    textures: Sequence[str] = ("photorealistic", "flat", "random_noise"),
    preferred_split: str = "test",
) -> list[Path]:
    """Create all requested qualitative task tables and a markdown index."""
    output_dir = Path(output_dir)
    paths: list[Path] = []
    for task in tasks:
        label_path = task_label_paths.get(task)
        if label_path is None or not Path(label_path).is_file():
            continue
        path = output_dir / f"qualitative_{_slug(task)}_{_slug(layer_name)}.png"
        try:
            paths.append(
                make_qualitative_probe_table(
                    task=task,
                    manifest_path=manifest_path,
                    label_path=label_path,
                    probe_root=probe_root,
                    output_path=path,
                    project_root=project_root,
                    model_display_order=model_display_order,
                    layer_name=layer_name,
                    textures=textures,
                    preferred_split=preferred_split,
                )
            )
        except (FileNotFoundError, ValueError, KeyError):
            continue
    if paths:
        index_path = output_dir / "qualitative_tables.md"
        lines = [
            "# Qualitative Probe Tables",
            "",
            (
                "These tables show one matched texture-control triplet. "
                "Model cells visualize final-layer linear-probe outputs, not "
                "dense decoder predictions."
            ),
            "",
        ]
        for path in paths:
            rel = path.name
            lines.extend([f"## {path.stem}", "", f"![{path.stem}]({rel})", ""])
        index_path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(index_path)
    return paths

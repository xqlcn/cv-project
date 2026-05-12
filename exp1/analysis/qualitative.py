"""Qualitative visual tables for Experiment 1 probe outputs."""

from __future__ import annotations

import os
import re
import tempfile
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


def _masked_vector_image(
    vector: Sequence[float],
    mask_path: Union[str, Path],
    project_root: Path,
    *,
    size: int,
) -> np.ndarray:
    mask = np.load(_resolve_path(project_root, mask_path), allow_pickle=False).astype(
        bool
    )
    vector_arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector_arr))
    if np.isfinite(norm) and norm > 1e-8:
        vector_arr = vector_arr / norm
    color = _normal_to_rgb(vector_arr.reshape(1, 1, 3))[0, 0]
    image = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    image[mask] = color
    return _fit_image(Image.fromarray(image), size=size)


def _surface_normal_ground_truth(
    label_row: pd.Series,
    manifest_row: Mapping[str, Any],
    project_root: Path,
    *,
    size: int,
) -> np.ndarray:
    vector = [
        label_row["mean_normal_x"],
        label_row["mean_normal_y"],
        label_row["mean_normal_z"],
    ]
    return _masked_vector_image(vector, manifest_row["mask_path"], project_root, size=size)


def _surface_normal_prediction(
    pred_row: pd.Series,
    manifest_row: Mapping[str, Any],
    project_root: Path,
    *,
    size: int,
) -> np.ndarray:
    vector = [
        pred_row["pred_mean_normal_x"],
        pred_row["pred_mean_normal_y"],
        pred_row["pred_mean_normal_z"],
    ]
    return _masked_vector_image(vector, manifest_row["mask_path"], project_root, size=size)


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


def _relative_depth_image(
    label_row: pd.Series,
    *,
    pred_row: Optional[pd.Series],
    size: int,
) -> np.ndarray:
    grid, valid = _relative_depth_scores(label_row, pred_row=pred_row)
    values = grid.copy()
    values[~valid] = np.nan
    if np.isfinite(values).any():
        fill = float(np.nanmean(values))
        values = np.nan_to_num(values, nan=fill)
    else:
        values = np.zeros_like(values)
    pixels = np.repeat(np.repeat(values, 32, axis=0), 32, axis=1)
    plt = _pyplot()
    cmap = plt.get_cmap("magma")
    rgba = cmap(np.clip(pixels, 0.0, 1.0))
    image = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return _fit_image(Image.fromarray(image), size=size)


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
    image_size: int = 144,
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
        figsize=(1.45 * len(columns), 1.65 * len(textures)),
        squeeze=False,
    )

    for row_idx, texture in enumerate(textures):
        manifest_row = triplet[texture]
        render_id = str(manifest_row["render_id"])
        label_row = labels_lookup[render_id]
        row_images: list[tuple[Optional[np.ndarray], Optional[str]]] = [
            (None, TEXTURE_DISPLAY_NAMES.get(texture, texture)),
            (_rgb_image(manifest_row, project_root, size=image_size), None),
        ]
        if task == "surface_normal_aggregate":
            row_images.append(
                (
                    _surface_normal_ground_truth(
                        label_row,
                        manifest_row,
                        project_root,
                        size=image_size,
                    ),
                    None,
                )
            )
        elif task == "relative_depth_regions":
            row_images.append(
                (
                    _relative_depth_image(label_row, pred_row=None, size=image_size),
                    None,
                )
            )
        else:
            row_images.append(_placeholder("unsupported", size=image_size))

        for model_name in model_names:
            pred_df = predictions.get((model_name, texture))
            if pred_df is None:
                row_images.append(_placeholder("not run", size=image_size))
                continue
            candidates = pred_df[
                (pred_df["render_id"].astype(str) == render_id)
                & (pred_df["split"].astype(str) == str(manifest_row["split"]))
            ]
            if candidates.empty:
                row_images.append(_placeholder("no row", size=image_size))
                continue
            pred_row = candidates.iloc[0]
            if task == "surface_normal_aggregate":
                row_images.append(
                    (
                        _surface_normal_prediction(
                            pred_row,
                            manifest_row,
                            project_root,
                            size=image_size,
                        ),
                        None,
                    )
                )
            elif task == "relative_depth_regions":
                row_images.append(
                    (
                        _relative_depth_image(
                            label_row,
                            pred_row=pred_row,
                            size=image_size,
                        ),
                        None,
                    )
                )
            else:
                row_images.append(_placeholder("unsupported", size=image_size))

        for col_idx, (image, text) in enumerate(row_images):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if image is not None:
                ax.imshow(image)
            if text:
                ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=9)

    title = task.replace("_", " ")
    fig.suptitle(f"{title} qualitative examples ({layer_name})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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

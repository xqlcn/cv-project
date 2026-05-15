"""Shared helpers for Experiment 1 rerender analysis scripts.

Centralizes:
- task / metric registries (with metric direction and chance baselines),
- pretty model/layer/texture display labels and colors,
- result-CSV and per-render predictions loaders,
- consistent matplotlib styling.

Designed to be imported by the scripts in ``scripts/analysis/`` and to keep
all directories configurable via CLI arguments (no hard-coded absolute paths).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------

MODEL_DISPLAY: Dict[str, str] = {
    "clip_vit_b16": "CLIP ViT-B/16",
    "clip_vit_l14": "CLIP ViT-L/14",
    "dinov2_vit_b": "DINOv2 ViT-B",
    "dinov2_vit_l": "DINOv2 ViT-L",
}

MODEL_FAMILY: Dict[str, str] = {
    "clip_vit_b16": "CLIP",
    "clip_vit_l14": "CLIP",
    "dinov2_vit_b": "DINOv2",
    "dinov2_vit_l": "DINOv2",
}

MODEL_COLORS: Dict[str, str] = {
    "clip_vit_b16": "#1f77b4",
    "clip_vit_l14": "#17becf",
    "dinov2_vit_b": "#d62728",
    "dinov2_vit_l": "#ff7f0e",
}

TEXTURE_DISPLAY: Dict[str, str] = {
    "photorealistic": "Photorealistic",
    "flat": "Flat Textureless",
    "random_noise": "Random Noise",
}

TEXTURE_COLORS: Dict[str, str] = {
    "photorealistic": "#2ca02c",
    "flat": "#9467bd",
    "random_noise": "#8c564b",
}

TEXTURE_ORDER: Sequence[str] = ("photorealistic", "flat", "random_noise")

LAYER_ORDER: Sequence[str] = ("layer4", "layer8", "layer12", "final")


def layer_sort_key(layer: str) -> Tuple[int, str]:
    """Return an int index for sorting layer names (final last)."""
    if layer == "final":
        return (99, layer)
    if layer.startswith("layer"):
        try:
            return (int(layer[5:]), layer)
        except ValueError:
            pass
    return (50, layer)


# ---------------------------------------------------------------------------
# Task / metric registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskInfo:
    """Per-task information needed for fair plotting and interpretation."""

    name: str
    pretty: str
    primary_metric: str  # name in the long results CSV
    direction: str       # "higher" or "lower" is better
    chance: Optional[float]  # baseline reference, None if unknown
    per_render_column: Optional[str]  # column in predictions.csv that is a per-render score
    per_render_direction: Optional[str]  # direction of the per_render_column
    family: str  # "main" or "dense"
    short: str  # short name for figure filenames


GLOBAL_TASKS: Dict[str, TaskInfo] = {
    "relative_depth_regions": TaskInfo(
        name="relative_depth_regions",
        pretty="Relative depth (region pairs)",
        primary_metric="valid_pair_accuracy",
        direction="higher",
        chance=0.5,
        per_render_column="row_valid_pair_accuracy",
        per_render_direction="higher",
        family="main",
        short="reldepth",
    ),
    "apparent_scale": TaskInfo(
        name="apparent_scale",
        pretty="Apparent scale (object + fg fraction)",
        primary_metric="mae_mean",
        direction="lower",
        chance=None,
        per_render_column="row_mae",
        per_render_direction="lower",
        family="main",
        short="scale",
    ),
    "lighting_direction": TaskInfo(
        name="lighting_direction",
        pretty="Lighting direction (unit vector)",
        primary_metric="angular_error_deg_mean",
        direction="lower",
        chance=90.0,  # mean angular error for random unit vectors
        per_render_column="angular_error_deg",
        per_render_direction="lower",
        family="main",
        short="lightdir",
    ),
    "lighting_intensity": TaskInfo(
        name="lighting_intensity",
        pretty="Lighting intensity (log)",
        primary_metric="mae_mean",
        direction="lower",
        chance=None,
        per_render_column="row_mae",
        per_render_direction="lower",
        family="main",
        short="lightint",
    ),
    "viewpoint": TaskInfo(
        name="viewpoint",
        pretty="Viewpoint (azimuth + elevation)",
        primary_metric="viewpoint_angular_error_deg_mean",
        direction="lower",
        chance=90.0,
        per_render_column="viewpoint_angular_error_deg",
        per_render_direction="lower",
        family="main",
        short="view",
    ),
}

DENSE_TASKS: Dict[str, TaskInfo] = {
    "dense_depth_patches": TaskInfo(
        name="dense_depth_patches",
        pretty="Dense depth (patch grid, SSI)",
        primary_metric="pearson_r_mean",
        direction="higher",
        chance=0.0,
        per_render_column="pearson_r",
        per_render_direction="higher",
        family="dense",
        short="densedepth",
    ),
    "dense_surface_normal_patches": TaskInfo(
        name="dense_surface_normal_patches",
        pretty="Dense surface normals (patch grid)",
        primary_metric="angular_error_deg_mean",
        direction="lower",
        chance=90.0,  # uniform random unit vectors
        per_render_column="angular_error_deg_mean",
        per_render_direction="lower",
        family="dense",
        short="densenorm",
    ),
}

ALL_TASKS: Dict[str, TaskInfo] = {**GLOBAL_TASKS, **DENSE_TASKS}


def texture_dir_name(task: TaskInfo, texture: str) -> str:
    """Return the per-texture probe subdirectory name (e.g. 'texture_flat' vs 'within_flat')."""
    if task.family == "dense":
        return f"within_{texture}"
    return f"texture_{texture}"


# ---------------------------------------------------------------------------
# Result CSV loading
# ---------------------------------------------------------------------------


def load_results(results_csv: Path) -> pd.DataFrame:
    """Load an ``exp1_results_long.csv``.

    Returns the long-format dataframe with a normalized ``texture_condition``
    column (string, with cross-texture combos preserved).
    """
    df = pd.read_csv(results_csv)
    df["texture_condition"] = df["texture_condition"].astype(str)
    return df


def filter_within_texture(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only same-train/eval texture rows (drops cross-texture combos)."""
    return df[df["texture_condition"].isin(TEXTURE_ORDER)].copy()


# ---------------------------------------------------------------------------
# Per-render predictions loaders
# ---------------------------------------------------------------------------


def find_probe_dir(probe_root: Path, model: str, layer: str, task: str, sub: str) -> Path:
    """Return path to ``probe_root/<model>/<layer>/<task>/<sub>``."""
    return probe_root / model / layer / task / sub


def load_predictions_csv(
    probe_root: Path,
    model: str,
    layer: str,
    task: str,
    texture: str,
    family: str,
) -> Optional[pd.DataFrame]:
    """Load the ``predictions.csv`` for a per-render probe run, if present."""
    task_info = ALL_TASKS.get(task)
    if task_info is None:
        return None
    sub = texture_dir_name(task_info, texture)
    path = find_probe_dir(probe_root, model, layer, task, sub) / "predictions.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df["model"] = model
    df["layer"] = layer
    df["task"] = task
    df["texture_condition"] = texture
    df["probe_family"] = family
    return df


def iter_per_render_predictions(
    probe_root: Path,
    family: str,
    tasks: Sequence[str],
    models: Optional[Sequence[str]] = None,
    layers: Optional[Sequence[str]] = None,
    textures: Sequence[str] = TEXTURE_ORDER,
) -> Iterable[pd.DataFrame]:
    """Yield per-render dataframes across (model, layer, task, texture)."""
    if not probe_root.is_dir():
        return
    model_dirs = sorted(probe_root.iterdir())
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if models and model not in models:
            continue
        for layer_dir in sorted(model_dir.iterdir(), key=lambda p: layer_sort_key(p.name)):
            if not layer_dir.is_dir():
                continue
            layer = layer_dir.name
            if layers and layer not in layers:
                continue
            for task in tasks:
                for tex in textures:
                    df = load_predictions_csv(probe_root, model, layer, task, tex, family)
                    if df is not None and not df.empty:
                        yield df


def load_per_render_table(
    probe_root: Path,
    family: str,
    tasks: Sequence[str],
    models: Optional[Sequence[str]] = None,
    layers: Optional[Sequence[str]] = None,
    textures: Sequence[str] = TEXTURE_ORDER,
) -> pd.DataFrame:
    """Concatenate all per-render predictions across (model, layer, task, texture)."""
    parts = list(
        iter_per_render_predictions(
            probe_root,
            family=family,
            tasks=tasks,
            models=models,
            layers=layers,
            textures=textures,
        )
    )
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Loading dense NPZ predictions
# ---------------------------------------------------------------------------


def load_dense_npz(
    probe_root: Path,
    model: str,
    layer: str,
    task: str,
    texture: str,
    split: str = "test",
) -> Optional[Mapping[str, np.ndarray]]:
    """Return the per-render predictions/targets/valid arrays for a dense probe."""
    task_info = DENSE_TASKS.get(task)
    if task_info is None:
        return None
    sub = texture_dir_name(task_info, texture)
    path = find_probe_dir(probe_root, model, layer, task, sub) / f"predictions_{split}.npz"
    if not path.is_file():
        return None
    return dict(np.load(path, allow_pickle=False))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def metric_better_is_higher(direction: str) -> bool:
    return direction.lower() == "higher"


def normalize_metric_for_heatmap(values: pd.Series, direction: str) -> pd.Series:
    """Return a 'higher-is-better' version of a metric for cross-task heatmaps.

    For "lower-is-better" metrics, returns ``-values``; for accuracies, returns
    the raw values. Used only for relative coloring across tasks.
    """
    if metric_better_is_higher(direction):
        return values
    return -values


def chance_line(task_info: TaskInfo) -> Optional[float]:
    return task_info.chance


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------


def get_pyplot():
    """Import matplotlib in a headless-safe way."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        fig.clf()
    except Exception:
        pass


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save the plot table as CSV next to the figure for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if manifest_path is None:
        return None
    p = Path(manifest_path)
    if not p.is_file():
        return None
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".jsonl",):
        return pd.read_json(p, lines=True)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported manifest format: {p.suffix}")


MANIFEST_STATE_COLUMNS: Tuple[str, ...] = (
    "camera_distance",
    "camera_azimuth_deg",
    "camera_elevation_deg",
    "light_azimuth_deg",
    "light_elevation_deg",
    "light_intensity",
    "object_scale",
    "qc_foreground_fraction",
    "category",
)


def merge_with_manifest(
    df: pd.DataFrame,
    manifest: Optional[pd.DataFrame],
    columns: Sequence[str] = MANIFEST_STATE_COLUMNS,
) -> pd.DataFrame:
    """Left-merge per-render df with manifest columns on ``render_id``."""
    if manifest is None:
        return df
    cols = ["render_id"] + [c for c in columns if c in manifest.columns]
    if "render_id" not in manifest.columns:
        return df
    keep_cols = [c for c in cols if c not in df.columns or c == "render_id"]
    return df.merge(manifest[keep_cols], on="render_id", how="left")

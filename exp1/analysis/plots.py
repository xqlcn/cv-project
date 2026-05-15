"""Matplotlib plots for Experiment 1 aggregated results."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

_CACHE_ROOT = Path(tempfile.gettempdir()) / "cv_project_plot_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_CACHE_ROOT / "matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib
import pandas as pd

from exp1.evaluation.comparisons import metric_direction


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def layer_sort_value(layer: object) -> int:
    """Sort intermediate ViT layers before the final representation."""
    text = str(layer).lower()
    if text.startswith("layer") and text.removeprefix("layer").isdigit():
        return int(text.removeprefix("layer"))
    if text == "final":
        return 10_000
    return 9_000


def _selected_splits(
    results: pd.DataFrame, splits: Optional[Iterable[str]]
) -> list[str]:
    available = [str(split) for split in results["split"].dropna().unique()]
    if splits is not None:
        return [str(split) for split in splits if str(split) in set(available)]
    return ["test"] if "test" in available else available


def _selected_metrics(
    results: pd.DataFrame, metrics: Optional[Iterable[str]]
) -> set[str]:
    if metrics is not None:
        return {str(metric) for metric in metrics}
    return {
        str(metric)
        for metric in results["metric"].dropna().unique()
        if metric_direction(str(metric)) is not None
    }


def plot_layerwise_metrics(
    results: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    splits: Optional[Iterable[str]] = None,
    metrics: Optional[Iterable[str]] = None,
) -> list[Path]:
    """Generate layer-wise metric plots grouped by task/split/texture/metric."""
    if results.empty:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = results.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    split_values = _selected_splits(df, splits)
    metric_values = _selected_metrics(df, metrics)
    df = df[
        df["split"].astype(str).isin(split_values)
        & df["metric"].astype(str).isin(metric_values)
    ]
    paths: list[Path] = []
    plt = _pyplot()
    group_cols = ["task", "split", "texture_condition", "metric"]
    for keys, group in df.groupby(group_cols, dropna=False):
        task, split, texture, metric = [str(value) for value in keys]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, model_rows in group.groupby("model"):
            rows = model_rows.sort_values(
                by="layer",
                key=lambda series: series.map(layer_sort_value),
            )
            ax.plot(
                rows["layer"].astype(str).tolist(),
                rows["value"].astype(float).tolist(),
                marker="o",
                linewidth=1.8,
                label=str(model),
            )
        ax.set_title(f"{task} | {split} | {texture}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = (
            output_dir
            / f"layerwise_{_slug(task)}_{_slug(split)}_{_slug(texture)}_{_slug(metric)}.png"
        )
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_texture_drops(
    drops: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    splits: Optional[Iterable[str]] = None,
) -> list[Path]:
    """Generate compact bar plots for texture-dependence drops."""
    if drops.empty:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = drops.copy()
    split_values = _selected_splits(df, splits)
    df = df[df["split"].astype(str).isin(split_values)]

    paths: list[Path] = []
    plt = _pyplot()
    for keys, group in df.groupby(["task", "split", "metric"], dropna=False):
        task, split, metric = [str(value) for value in keys]
        group = group.sort_values(["model", "layer", "comparison_texture"])
        labels = [
            f"{row.model}\n{row.layer}\n{row.comparison_texture}"
            for row in group.itertuples()
        ]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.55), 4.5))
        ax.bar(range(len(labels)), group["texture_drop"].astype(float).tolist())
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Texture drop (positive = worse)")
        ax.set_title(f"{task} | {split} | {metric}")
        fig.tight_layout()
        path = (
            output_dir
            / f"texture_drop_{_slug(task)}_{_slug(split)}_{_slug(metric)}.png"
        )
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths

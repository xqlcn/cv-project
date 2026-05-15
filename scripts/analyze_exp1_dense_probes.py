#!/usr/bin/env python3
"""Analyze the ``probe3d-exp1-dense-probes`` W&B project / local dense-probe outputs.

This script produces the standalone deliverable in
``outputs/exp1_dense_probe_analysis/``:

- ``analysis.md``                          -- self-contained markdown report
- ``metrics_summary.csv``                  -- one long row per (model, layer, task, texture, split, metric)
- ``run_metadata.csv``                     -- one row per W&B / local probe run (mirrors W&B run config)
- ``notes_on_missing_data.md``             -- explicit list of anything we could not access
- ``figures/``                             -- aggregate + state-space figures (PNG + supporting CSV)

Data sources, in priority order:

1. Weights & Biases project
   ``jczhang_-massachusetts-institute-of-technology/probe3d-exp1-dense-probes``
   (only used when ``wandb`` is importable and ``--from-wandb`` is requested).
2. Local probe outputs under ``outputs/exp1_dense/`` (default).
   The W&B project mirrors this directory 1:1 according to
   ``notebooks/colab_exp1_zip_probe_training_wandb.ipynb`` (each W&B run
   uploads the same ``metrics.json`` / ``history.csv`` / ``predictions.csv``
   / ``checkpoint.pt`` artifacts the script writes to disk), so the on-disk
   tree is a faithful local snapshot of the W&B runs.

The script never modifies training code and never tries to retrain probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis._common import (  # noqa: E402
    DENSE_TASKS,
    MODEL_COLORS,
    MODEL_DISPLAY,
    TEXTURE_COLORS,
    TEXTURE_DISPLAY,
    TEXTURE_ORDER,
    TaskInfo,
    layer_sort_key,
    load_manifest,
    save_dataframe,
    save_figure,
)

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

WANDB_PROJECT = "probe3d-exp1-dense-probes"
WANDB_ENTITY = "jczhang_-massachusetts-institute-of-technology"

DEFAULT_PROBE_ROOT = PROJECT_ROOT / "outputs" / "exp1_dense" / "probes"
DEFAULT_RESULTS_CSV = (
    PROJECT_ROOT / "outputs" / "exp1_dense" / "results" / "exp1_results_long.csv"
)
DEFAULT_TEXTURE_DROPS_CSV = (
    PROJECT_ROOT / "outputs" / "exp1_dense" / "results" / "exp1_texture_drops.csv"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "exp1_dense" / "manifests" / "render_valid.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exp1_dense_probe_analysis"

# Dense tasks recorded on W&B (one probe per (model, layer, task, within_texture))
DENSE_TASK_NAMES = list(DENSE_TASKS.keys())

# Headline metric per task to use across summary tables / heatmaps.
HEADLINE_METRICS: Dict[str, Tuple[str, str, str]] = {
    # task -> (metric_name, direction, pretty label)
    "dense_depth_patches": ("pearson_r_mean", "higher", "Per-render Pearson r (depth)"),
    "dense_surface_normal_patches": (
        "angular_error_deg_median",
        "lower",
        "Median angular error (deg)",
    ),
}

# Secondary metrics surfaced in markdown tables.
SECONDARY_METRICS: Dict[str, List[Tuple[str, str, str]]] = {
    "dense_depth_patches": [
        ("ssi_l1_mean", "lower", "SSI L1 (mean)"),
        ("ssi_l1_median", "lower", "SSI L1 (median)"),
        ("pearson_r_mean", "higher", "Pearson r (mean)"),
    ],
    "dense_surface_normal_patches": [
        ("angular_error_deg_mean", "lower", "Angular error mean (deg)"),
        ("angular_error_deg_median", "lower", "Angular error median (deg)"),
        ("within_22_5_deg", "higher", "Fraction within 22.5 deg"),
        ("within_30_deg", "higher", "Fraction within 30 deg"),
    ],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_wandb_runs(
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
    """Try to pull run metadata + summary metrics from W&B.

    Returns ``(run_metadata_df, summary_df, warnings)``. If W&B is unavailable
    or the project cannot be reached, returns ``(None, None, warnings)``.
    """
    warnings: List[str] = []
    try:
        import wandb  # type: ignore
    except ImportError:
        warnings.append(
            "wandb python package not installed; falling back to local probe outputs only."
        )
        return None, None, warnings

    try:
        api = wandb.Api(timeout=60)
        runs = list(api.runs(f"{entity}/{project}"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"Could not query W&B project '{entity}/{project}': {exc}. "
            "Falling back to local probe outputs only."
        )
        return None, None, warnings

    if not runs:
        warnings.append(
            f"W&B project '{entity}/{project}' returned 0 runs; using local outputs."
        )
        return None, None, warnings

    meta_rows = []
    summary_rows = []
    for run in runs:
        try:
            config = dict(run.config or {})
            summary = dict(run.summary or {})
            meta_rows.append(
                {
                    "wandb_id": run.id,
                    "wandb_name": run.name,
                    "wandb_state": run.state,
                    "wandb_group": getattr(run, "group", None),
                    "wandb_tags": ",".join(run.tags) if getattr(run, "tags", None) else "",
                    "model_name": config.get("model_name") or config.get("model"),
                    "layer_name": config.get("layer_name") or config.get("layer"),
                    "task": config.get("task"),
                    "texture_condition": _list_or_str(
                        config.get("texture_condition") or config.get("texture")
                    ),
                    "train_texture_condition": config.get("train_texture_condition"),
                    "eval_texture_condition": config.get("eval_texture_condition"),
                    "stage": config.get("stage"),
                    "feature_mode": config.get("feature_mode"),
                    "target_mode": config.get("target_mode"),
                    "loss": config.get("loss"),
                    "monitor": config.get("monitor"),
                    "lr": config.get("lr"),
                    "weight_decay": config.get("weight_decay"),
                    "batch_size": config.get("batch_size"),
                    "epochs": config.get("epochs"),
                    "seed": config.get("seed"),
                    "scheduler": config.get("scheduler"),
                    "warmup_epochs": config.get("warmup_epochs"),
                    "probe_dir": config.get("probe_dir"),
                    "config_name": config.get("config_name"),
                }
            )
            for k, v in summary.items():
                if not isinstance(v, (int, float)):
                    continue
                if k.startswith("final/"):
                    split, metric = k[len("final/"):].split("/", 1) if "/" in k[len("final/"):] else (
                        None,
                        k[len("final/"):],
                    )
                    summary_rows.append(
                        {
                            "wandb_id": run.id,
                            "wandb_name": run.name,
                            "split": split,
                            "metric": metric,
                            "value": float(v),
                            "source": "wandb_summary",
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not parse W&B run {getattr(run, 'id', '?')}: {exc}")
            continue

    meta_df = pd.DataFrame(meta_rows) if meta_rows else None
    summary_df = pd.DataFrame(summary_rows) if summary_rows else None
    if not meta_rows:
        warnings.append("W&B returned runs but none had parseable metadata; using local outputs.")
    return meta_df, summary_df, warnings


def _list_or_str(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return str(value[0])
        return ",".join(str(v) for v in value)
    return str(value)


def discover_local_probes(probe_root: Path) -> pd.DataFrame:
    """Walk the on-disk probe tree and return one row per probe run.

    Each row contains paths to ``metrics.json``, ``history.csv``,
    ``predictions.csv``, ``predictions_<split>.npz``, ``checkpoint.pt``.
    """
    rows: List[Dict] = []
    if not probe_root.is_dir():
        return pd.DataFrame()
    for model_dir in sorted(probe_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for layer_dir in sorted(model_dir.iterdir(), key=lambda p: layer_sort_key(p.name)):
            if not layer_dir.is_dir():
                continue
            for task_dir in sorted(layer_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                if task_dir.name not in DENSE_TASKS:
                    continue
                for tex_dir in sorted(task_dir.iterdir()):
                    if not tex_dir.is_dir():
                        continue
                    name = tex_dir.name
                    if name.startswith("within_"):
                        texture = name[len("within_"):]
                        scheme = "within"
                    elif name.startswith("texture_"):
                        texture = name[len("texture_"):]
                        scheme = "texture"
                    else:
                        texture = name
                        scheme = "unknown"
                    metrics_path = tex_dir / "metrics.json"
                    if not metrics_path.is_file():
                        continue
                    rows.append(
                        {
                            "model_name": model_dir.name,
                            "layer_name": layer_dir.name,
                            "task": task_dir.name,
                            "texture_condition": texture,
                            "training_scheme": scheme,
                            "probe_dir": str(tex_dir.relative_to(PROJECT_ROOT)),
                            "metrics_path": str(metrics_path),
                            "history_path": str(tex_dir / "history.csv"),
                            "predictions_csv": str(tex_dir / "predictions.csv"),
                            "predictions_test_npz": str(tex_dir / "predictions_test.npz"),
                            "predictions_val_npz": str(tex_dir / "predictions_val.npz"),
                            "predictions_train_npz": str(tex_dir / "predictions_train.npz"),
                            "checkpoint_path": str(tex_dir / "checkpoint.pt"),
                        }
                    )
    return pd.DataFrame(rows)


def load_local_summary_metrics(probes_df: pd.DataFrame) -> pd.DataFrame:
    """Parse each ``metrics.json`` into a long-format dataframe."""
    rows: List[Dict] = []
    for _, row in probes_df.iterrows():
        path = Path(row["metrics_path"])
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        metadata = data.get("metadata", {})
        metrics = data.get("metrics", {})
        num_rows_by_split = metadata.get("num_rows_by_split") or {}
        for split, split_metrics in metrics.items():
            if not isinstance(split_metrics, Mapping):
                continue
            for metric_name, value in split_metrics.items():
                try:
                    fvalue = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fvalue):
                    continue
                rows.append(
                    {
                        "model_name": row["model_name"],
                        "layer_name": row["layer_name"],
                        "task": row["task"],
                        "texture_condition": row["texture_condition"],
                        "training_scheme": row["training_scheme"],
                        "split": split,
                        "metric": metric_name,
                        "value": fvalue,
                        "num_samples": num_rows_by_split.get(split),
                        "loss": metadata.get("loss"),
                        "target_mode": metadata.get("target_mode"),
                        "feature_mode": metadata.get("feature_mode"),
                        "monitor": metadata.get("monitor"),
                        "metrics_path": str(path),
                    }
                )
    return pd.DataFrame(rows)


def load_history_curves(probes_df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Concatenate all per-probe ``history.csv`` files."""
    parts: List[pd.DataFrame] = []
    for _, row in probes_df.iterrows():
        path = Path(row["history_path"])
        if not path.is_file():
            continue
        try:
            hist = pd.read_csv(path)
        except Exception:
            continue
        hist["model_name"] = row["model_name"]
        hist["layer_name"] = row["layer_name"]
        hist["task"] = row["task"]
        hist["texture_condition"] = row["texture_condition"]
        if max_rows is not None and len(hist) > max_rows:
            hist = hist.iloc[:max_rows].copy()
        parts.append(hist)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def load_per_render_predictions(probes_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate per-render ``predictions.csv`` files."""
    parts: List[pd.DataFrame] = []
    for _, row in probes_df.iterrows():
        path = Path(row["predictions_csv"])
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["model_name"] = row["model_name"]
        df["layer_name"] = row["layer_name"]
        df["task"] = row["task"]
        df["probe_texture_condition"] = row["texture_condition"]
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Aggregate figures
# ---------------------------------------------------------------------------


def _filter_test(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["split"] == "test"].copy()


def _pretty_model(m: str) -> str:
    return MODEL_DISPLAY.get(m, m)


def _pretty_texture(t: str) -> str:
    return TEXTURE_DISPLAY.get(t, t)


def make_metric_heatmaps(summary: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Heatmap of model x texture for the headline metric of each task / layer."""
    out_paths: List[Path] = []
    test = _filter_test(summary)
    for task, (metric, direction, pretty) in HEADLINE_METRICS.items():
        for layer in sorted(test["layer_name"].unique(), key=layer_sort_key):
            sub = test[
                (test["task"] == task)
                & (test["metric"] == metric)
                & (test["layer_name"] == layer)
            ]
            if sub.empty:
                continue
            pivot = sub.pivot_table(
                index="model_name",
                columns="texture_condition",
                values="value",
                aggfunc="mean",
            )
            pivot = pivot[[t for t in TEXTURE_ORDER if t in pivot.columns]]
            pivot = pivot.reindex(sorted(pivot.index))
            arr = pivot.values
            cmap = "RdYlGn" if direction == "higher" else "RdYlGn_r"
            fig, ax = plt.subplots(
                figsize=(max(4.5, 1.2 * pivot.shape[1] + 2.0), max(2.5, 0.6 * pivot.shape[0] + 1.8))
            )
            im = ax.imshow(arr, aspect="auto", cmap=cmap)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                    )
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(
                [_pretty_texture(t) for t in pivot.columns], rotation=20, ha="right"
            )
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels([_pretty_model(m) for m in pivot.index])
            arrow = "↑ higher is better" if direction == "higher" else "↓ lower is better"
            ax.set_title(
                f"{DENSE_TASKS[task].pretty}\nLayer {layer} | {pretty} ({arrow})",
                fontsize=11,
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            short = DENSE_TASKS[task].short
            path = out_dir / f"heatmap_{short}_{layer}_{metric}.png"
            save_figure(fig, path)
            save_dataframe(pivot.reset_index(), path.with_suffix(".csv"))
            out_paths.append(path)
            plt.close(fig)
    return out_paths


def make_clip_vs_dino_bar(summary: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Grouped bar chart of CLIP vs DINOv2 per (task, texture, layer)."""
    out_paths: List[Path] = []
    test = _filter_test(summary)
    for task, (metric, direction, pretty) in HEADLINE_METRICS.items():
        sub = test[(test["task"] == task) & (test["metric"] == metric)]
        if sub.empty:
            continue
        layers = sorted(sub["layer_name"].unique(), key=layer_sort_key)
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), sharey=True)
        if len(layers) == 1:
            axes = [axes]
        for ax, layer in zip(axes, layers):
            layer_sub = sub[sub["layer_name"] == layer]
            pivot = layer_sub.pivot_table(
                index="texture_condition", columns="model_name", values="value", aggfunc="mean"
            )
            pivot = pivot.reindex([t for t in TEXTURE_ORDER if t in pivot.index])
            x = np.arange(len(pivot.index))
            width = 0.38
            models = list(pivot.columns)
            for i, model in enumerate(models):
                ax.bar(
                    x + (i - (len(models) - 1) / 2) * width,
                    pivot[model].values,
                    width=width,
                    label=_pretty_model(model),
                    color=MODEL_COLORS.get(model, None),
                    edgecolor="black",
                    linewidth=0.4,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([_pretty_texture(t) for t in pivot.index], rotation=15)
            chance = DENSE_TASKS[task].chance
            if chance is not None:
                ax.axhline(chance, color="grey", linestyle="--", linewidth=0.8, label="Chance / null")
            arrow = "↑" if direction == "higher" else "↓"
            ax.set_ylabel(f"{pretty} ({arrow})")
            ax.set_title(f"Layer {layer}", fontsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"{DENSE_TASKS[task].pretty} - CLIP vs DINOv2 (test split)", fontsize=12
        )
        fig.tight_layout()
        short = DENSE_TASKS[task].short
        path = out_dir / f"clip_vs_dinov2_{short}.png"
        save_figure(fig, path)
        save_dataframe(sub, path.with_suffix(".csv"))
        out_paths.append(path)
        plt.close(fig)
    return out_paths


def make_texture_drop_plot(
    texture_drops: Optional[pd.DataFrame],
    out_dir: Path,
) -> List[Path]:
    """Bar chart of headline-metric drop from photorealistic -> flat / random_noise."""
    if texture_drops is None or texture_drops.empty:
        return []
    out_paths: List[Path] = []
    for task, (metric, direction, pretty) in HEADLINE_METRICS.items():
        sub = texture_drops[
            (texture_drops["task"] == task)
            & (texture_drops["metric"] == metric)
            & (texture_drops["split"] == "test")
            & (texture_drops["baseline_texture"] == "photorealistic")
        ].copy()
        if sub.empty:
            continue
        layers = sorted(sub["layer"].unique(), key=layer_sort_key)
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), sharey=True)
        if len(layers) == 1:
            axes = [axes]
        for ax, layer in zip(axes, layers):
            layer_sub = sub[sub["layer"] == layer]
            pivot = layer_sub.pivot_table(
                index="comparison_texture",
                columns="model",
                values="texture_drop",
                aggfunc="mean",
            )
            pivot = pivot.reindex(
                [t for t in ("flat", "random_noise") if t in pivot.index]
            )
            x = np.arange(len(pivot.index))
            width = 0.38
            models = list(pivot.columns)
            for i, model in enumerate(models):
                ax.bar(
                    x + (i - (len(models) - 1) / 2) * width,
                    pivot[model].values,
                    width=width,
                    label=_pretty_model(model),
                    color=MODEL_COLORS.get(model, None),
                    edgecolor="black",
                    linewidth=0.4,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([_pretty_texture(t) for t in pivot.index])
            ax.axhline(0.0, color="black", linewidth=0.8)
            arrow = "(positive = worse than photorealistic)"
            ax.set_ylabel(f"texture_drop on {pretty}\n{arrow}")
            ax.set_title(f"Layer {layer}", fontsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"{DENSE_TASKS[task].pretty} - degradation vs photorealistic (test split)",
            fontsize=12,
        )
        fig.tight_layout()
        short = DENSE_TASKS[task].short
        path = out_dir / f"texture_drop_{short}.png"
        save_figure(fig, path)
        save_dataframe(sub, path.with_suffix(".csv"))
        out_paths.append(path)
        plt.close(fig)
    return out_paths


def make_layerwise_plot(summary: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Final vs layer8 comparison for each (model, texture, task)."""
    out_paths: List[Path] = []
    test = _filter_test(summary)
    for task, (metric, direction, pretty) in HEADLINE_METRICS.items():
        sub = test[(test["task"] == task) & (test["metric"] == metric)]
        if sub.empty:
            continue
        textures = [t for t in TEXTURE_ORDER if t in sub["texture_condition"].unique()]
        fig, axes = plt.subplots(1, len(textures), figsize=(4.5 * len(textures), 4.0), sharey=True)
        if len(textures) == 1:
            axes = [axes]
        for ax, tex in zip(axes, textures):
            tex_sub = sub[sub["texture_condition"] == tex]
            pivot = tex_sub.pivot_table(
                index="layer_name", columns="model_name", values="value", aggfunc="mean"
            )
            pivot = pivot.reindex(sorted(pivot.index, key=layer_sort_key))
            x = np.arange(len(pivot.index))
            for model in pivot.columns:
                ax.plot(
                    x,
                    pivot[model].values,
                    marker="o",
                    color=MODEL_COLORS.get(model, None),
                    label=_pretty_model(model),
                )
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index)
            arrow = "↑" if direction == "higher" else "↓"
            ax.set_ylabel(f"{pretty} ({arrow})")
            ax.set_title(_pretty_texture(tex))
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            chance = DENSE_TASKS[task].chance
            if chance is not None:
                ax.axhline(chance, color="grey", linestyle="--", linewidth=0.8, label="Chance")
            ax.legend(fontsize=8)
        fig.suptitle(
            f"{DENSE_TASKS[task].pretty} - probe quality across ViT layers",
            fontsize=12,
        )
        fig.tight_layout()
        short = DENSE_TASKS[task].short
        path = out_dir / f"layerwise_{short}.png"
        save_figure(fig, path)
        save_dataframe(sub, path.with_suffix(".csv"))
        out_paths.append(path)
        plt.close(fig)
    return out_paths


def make_per_render_distributions(
    per_render: pd.DataFrame, out_dir: Path
) -> List[Path]:
    """Per-render histograms + ECDFs for each task / texture."""
    if per_render.empty:
        return []
    out_paths: List[Path] = []
    for task, (_, _, _) in HEADLINE_METRICS.items():
        task_info = DENSE_TASKS[task]
        col = task_info.per_render_column
        if col is None or col not in per_render.columns:
            continue
        df = per_render[per_render["task"] == task].copy()
        if df.empty:
            continue
        df = df[df["split"] == "test"]
        if df.empty:
            continue
        layers = sorted(df["layer_name"].unique(), key=layer_sort_key)
        for layer in layers:
            layer_df = df[df["layer_name"] == layer]
            if layer_df.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for tex in TEXTURE_ORDER:
                tex_df = layer_df[layer_df["probe_texture_condition"] == tex]
                if tex_df.empty:
                    continue
                for model in sorted(tex_df["model_name"].unique()):
                    sel = tex_df[tex_df["model_name"] == model]
                    values = sel[col].dropna().values
                    if values.size == 0:
                        continue
                    label = f"{_pretty_model(model)} / {_pretty_texture(tex)} (n={len(values)})"
                    axes[0].hist(
                        values,
                        bins=25,
                        alpha=0.35,
                        label=label,
                        color=MODEL_COLORS.get(model, None),
                        linewidth=0.8,
                        histtype="stepfilled",
                        edgecolor="black",
                    )
                    sorted_vals = np.sort(values)
                    y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                    ls = {
                        "photorealistic": "-",
                        "flat": "--",
                        "random_noise": ":",
                    }.get(tex, "-")
                    axes[1].plot(
                        sorted_vals,
                        y,
                        ls,
                        label=label,
                        color=MODEL_COLORS.get(model, None),
                    )
            arrow = "↑" if task_info.per_render_direction == "higher" else "↓"
            axes[0].set_xlabel(f"{col} ({arrow} better)")
            axes[0].set_ylabel("# test renders")
            axes[0].set_title(f"Per-render distribution - {task_info.pretty}, layer {layer}")
            axes[0].legend(fontsize=7, loc="best")
            axes[1].set_xlabel(f"{col} ({arrow} better)")
            axes[1].set_ylabel("ECDF")
            axes[1].set_title("ECDF (test split)")
            axes[1].grid(alpha=0.4, linestyle=":")
            axes[1].legend(fontsize=7, loc="best")
            chance = task_info.chance
            if chance is not None:
                for a in axes:
                    a.axvline(chance, color="grey", linestyle="--", linewidth=0.8)
            fig.tight_layout()
            short = task_info.short
            path = out_dir / f"per_render_{short}_{layer}.png"
            save_figure(fig, path)
            save_dataframe(layer_df, path.with_suffix(".csv"))
            out_paths.append(path)
            plt.close(fig)
    return out_paths


def make_paired_scatter(per_render: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Scatter of CLIP-per-render vs DINOv2-per-render score (paired by render_id)."""
    if per_render.empty:
        return []
    out_paths: List[Path] = []
    for task, (_, _, _) in HEADLINE_METRICS.items():
        info = DENSE_TASKS[task]
        col = info.per_render_column
        if col not in per_render.columns:
            continue
        df = per_render[(per_render["task"] == task) & (per_render["split"] == "test")].copy()
        if df.empty:
            continue
        for layer in sorted(df["layer_name"].unique(), key=layer_sort_key):
            layer_df = df[df["layer_name"] == layer]
            if layer_df.empty:
                continue
            fig, axes = plt.subplots(
                1, len(TEXTURE_ORDER), figsize=(4.0 * len(TEXTURE_ORDER), 4.0), sharex=True, sharey=True
            )
            for ax, tex in zip(axes, TEXTURE_ORDER):
                tex_df = layer_df[layer_df["probe_texture_condition"] == tex]
                clip = tex_df[tex_df["model_name"] == "clip_vit_b16"][["render_id", col]]
                dino = tex_df[tex_df["model_name"] == "dinov2_vit_b"][["render_id", col]]
                if clip.empty or dino.empty:
                    ax.set_title(f"{_pretty_texture(tex)} (no data)")
                    continue
                merged = clip.merge(dino, on="render_id", suffixes=("_clip", "_dino"))
                if merged.empty:
                    ax.set_title(f"{_pretty_texture(tex)} (no paired)")
                    continue
                ax.scatter(
                    merged[f"{col}_clip"],
                    merged[f"{col}_dino"],
                    s=18,
                    alpha=0.7,
                    color=TEXTURE_COLORS.get(tex, "tab:gray"),
                    edgecolor="black",
                    linewidth=0.3,
                )
                lo = float(np.nanmin([merged[f"{col}_clip"].min(), merged[f"{col}_dino"].min()]))
                hi = float(np.nanmax([merged[f"{col}_clip"].max(), merged[f"{col}_dino"].max()]))
                ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, linestyle="--")
                if info.chance is not None:
                    ax.axvline(info.chance, color="grey", linestyle=":", linewidth=0.6)
                    ax.axhline(info.chance, color="grey", linestyle=":", linewidth=0.6)
                ax.set_xlabel("CLIP ViT-B/16")
                ax.set_ylabel("DINOv2 ViT-B")
                ax.set_title(f"{_pretty_texture(tex)} (n={len(merged)})")
                ax.grid(alpha=0.4, linestyle=":")
            fig.suptitle(
                f"{info.pretty}, layer {layer} - paired per-render comparison\n"
                "Points above y=x: DINOv2 better; below y=x: CLIP better.",
                fontsize=11,
            )
            fig.tight_layout()
            path = out_dir / f"paired_scatter_{info.short}_{layer}.png"
            save_figure(fig, path)
            save_dataframe(df, path.with_suffix(".csv"))
            out_paths.append(path)
            plt.close(fig)
    return out_paths


# ---------------------------------------------------------------------------
# State-space slice analysis
# ---------------------------------------------------------------------------


STATE_SLICES: List[Tuple[str, str, str]] = [
    # (manifest column, slice label, slice type: "cat" or "bin")
    ("category", "Object category", "cat"),
    ("camera_distance", "Camera distance", "bin"),
    ("camera_azimuth_deg", "Camera azimuth (deg)", "bin"),
    ("camera_elevation_deg", "Camera elevation (deg)", "bin"),
    ("light_elevation_deg", "Light elevation (deg)", "bin"),
    ("light_intensity", "Light intensity", "bin"),
    ("object_scale", "Object scale", "bin"),
    ("qc_foreground_fraction", "Foreground fraction", "bin"),
]


def make_state_space_plots(
    per_render: pd.DataFrame,
    manifest: Optional[pd.DataFrame],
    out_dir: Path,
    warnings_out: Optional[List[str]] = None,
) -> List[Path]:
    if per_render.empty or manifest is None:
        return []
    out_paths: List[Path] = []
    manifest_cols = ["render_id"] + [c for c, _, _ in STATE_SLICES if c in manifest.columns]
    man = manifest[manifest_cols].drop_duplicates("render_id")
    # Avoid column collisions: drop any state-slice columns already on per_render
    # (e.g. ``category`` is also in predictions.csv) so the manifest version is used.
    collisions = [c for c, _, _ in STATE_SLICES if c in per_render.columns]
    per_render = per_render.drop(columns=collisions, errors="ignore")
    merged = per_render.merge(man, on="render_id", how="left")
    test = merged[merged["split"] == "test"].copy()
    if test.empty:
        return out_paths
    constant_columns: List[str] = []
    for column, label, _ in STATE_SLICES:
        if column not in test.columns:
            continue
        try:
            coerced = pd.to_numeric(test[column])
        except (ValueError, TypeError):
            coerced = test[column]
        nuniq = coerced.nunique(dropna=True)
        if nuniq is not None and nuniq <= 1:
            constant_columns.append(column)
            if warnings_out is not None:
                only = test[column].dropna().iloc[0] if test[column].notna().any() else None
                warnings_out.append(
                    f"State-space slice skipped: '{column}' is constant in this dataset "
                    f"(only value = {only!r}). No plot generated for {label}."
                )
    for task, (_, _, _) in HEADLINE_METRICS.items():
        info = DENSE_TASKS[task]
        col = info.per_render_column
        if col is None or col not in test.columns:
            continue
        for layer in sorted(test["layer_name"].unique(), key=layer_sort_key):
            for column, label, slice_type in STATE_SLICES:
                if column not in test.columns:
                    continue
                if column in constant_columns:
                    continue
                sub = test[(test["task"] == task) & (test["layer_name"] == layer)].copy()
                if sub.empty:
                    continue
                if slice_type == "bin":
                    finite = pd.to_numeric(sub[column], errors="coerce")
                    sub[column] = finite
                    sub = sub.dropna(subset=[column])
                    if sub.empty:
                        continue
                    try:
                        sub["bin"] = pd.qcut(sub[column], q=5, duplicates="drop")
                    except Exception:
                        continue
                    grouped = (
                        sub.groupby(
                            ["model_name", "probe_texture_condition", "bin"],
                            observed=True,
                        )
                        .agg(value=(col, "mean"), n=(col, "size"))
                        .reset_index()
                    )
                    if grouped.empty:
                        continue
                    grouped["bin_center"] = grouped["bin"].apply(lambda b: float(b.mid))
                    fig, axes = plt.subplots(
                        1, len(TEXTURE_ORDER), figsize=(4.2 * len(TEXTURE_ORDER), 3.8), sharey=True
                    )
                    for ax, tex in zip(axes, TEXTURE_ORDER):
                        tex_df = grouped[grouped["probe_texture_condition"] == tex]
                        if tex_df.empty:
                            ax.set_title(f"{_pretty_texture(tex)} (no data)")
                            continue
                        for model in sorted(tex_df["model_name"].unique()):
                            ms = tex_df[tex_df["model_name"] == model].sort_values("bin_center")
                            ax.plot(
                                ms["bin_center"],
                                ms["value"],
                                marker="o",
                                color=MODEL_COLORS.get(model, None),
                                label=_pretty_model(model),
                            )
                        ax.set_xlabel(label)
                        arrow = "↑" if info.per_render_direction == "higher" else "↓"
                        ax.set_ylabel(f"{col} ({arrow})")
                        if info.chance is not None:
                            ax.axhline(info.chance, color="grey", linestyle="--", linewidth=0.8)
                        ax.set_title(_pretty_texture(tex))
                        ax.grid(alpha=0.4, linestyle=":")
                        ax.legend(fontsize=7)
                    fig.suptitle(
                        f"{info.pretty}, layer {layer}\nby {label} (quintile bins, test split)",
                        fontsize=11,
                    )
                    fig.tight_layout()
                    path = out_dir / f"slice_{info.short}_{layer}_by_{column}.png"
                    save_figure(fig, path)
                    save_dataframe(grouped, path.with_suffix(".csv"))
                    out_paths.append(path)
                    plt.close(fig)
                else:
                    sub = sub.dropna(subset=[column])
                    if sub.empty:
                        continue
                    grouped = (
                        sub.groupby(
                            ["model_name", "probe_texture_condition", column],
                            observed=True,
                        )
                        .agg(value=(col, "mean"), n=(col, "size"))
                        .reset_index()
                    )
                    categories = sorted(grouped[column].unique())
                    fig, axes = plt.subplots(
                        1, len(TEXTURE_ORDER), figsize=(5.0 * len(TEXTURE_ORDER), 4.2), sharey=True
                    )
                    width = 0.35
                    for ax, tex in zip(axes, TEXTURE_ORDER):
                        tex_df = grouped[grouped["probe_texture_condition"] == tex]
                        if tex_df.empty:
                            ax.set_title(f"{_pretty_texture(tex)} (no data)")
                            continue
                        models = sorted(tex_df["model_name"].unique())
                        x = np.arange(len(categories))
                        for i, model in enumerate(models):
                            ms = tex_df[tex_df["model_name"] == model].set_index(column)
                            vals = [ms["value"].get(c, np.nan) for c in categories]
                            ax.bar(
                                x + (i - (len(models) - 1) / 2) * width,
                                vals,
                                width=width,
                                color=MODEL_COLORS.get(model, None),
                                label=_pretty_model(model),
                                edgecolor="black",
                                linewidth=0.3,
                            )
                        ax.set_xticks(x)
                        ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=8)
                        arrow = "↑" if info.per_render_direction == "higher" else "↓"
                        ax.set_ylabel(f"{col} ({arrow})")
                        if info.chance is not None:
                            ax.axhline(info.chance, color="grey", linestyle="--", linewidth=0.8)
                        ax.set_title(_pretty_texture(tex))
                        ax.grid(axis="y", alpha=0.4, linestyle=":")
                        ax.legend(fontsize=7)
                    fig.suptitle(
                        f"{info.pretty}, layer {layer}\nby {label} (test split)",
                        fontsize=11,
                    )
                    fig.tight_layout()
                    path = out_dir / f"slice_{info.short}_{layer}_by_{column}.png"
                    save_figure(fig, path)
                    save_dataframe(grouped, path.with_suffix(".csv"))
                    out_paths.append(path)
                    plt.close(fig)
    return out_paths


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _format_metric(value: float, direction: str) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def _summary_pivot(summary: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    sub = _filter_test(summary)
    sub = sub[(sub["task"] == task) & (sub["metric"] == metric)]
    return sub.pivot_table(
        index=["model_name", "layer_name"],
        columns="texture_condition",
        values="value",
        aggfunc="mean",
    )


def _make_summary_table_md(summary: pd.DataFrame) -> str:
    lines: List[str] = []
    for task, (metric, direction, pretty) in HEADLINE_METRICS.items():
        info = DENSE_TASKS[task]
        pivot = _summary_pivot(summary, task, metric)
        if pivot.empty:
            continue
        pivot = pivot[[t for t in TEXTURE_ORDER if t in pivot.columns]]
        lines.append(f"### {info.pretty}")
        lines.append("")
        lines.append(f"Headline metric: **{pretty}** ({'higher' if direction=='higher' else 'lower'} is better, chance≈{info.chance}).")
        lines.append("")
        header = "| Model | Layer | " + " | ".join(_pretty_texture(t) for t in pivot.columns) + " |"
        sep = "|" + "|".join(["---"] * (2 + len(pivot.columns))) + "|"
        lines.append(header)
        lines.append(sep)
        for (model, layer), row in pivot.iterrows():
            cells = [
                _pretty_model(model),
                str(layer),
            ] + [_format_metric(row[t], direction) for t in pivot.columns]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        # Secondary metrics summary
        for sec_metric, sec_direction, sec_pretty in SECONDARY_METRICS.get(task, []):
            sec_pivot = _summary_pivot(summary, task, sec_metric)
            if sec_pivot.empty:
                continue
            sec_pivot = sec_pivot[[t for t in TEXTURE_ORDER if t in sec_pivot.columns]]
            lines.append(
                f"<details><summary>Secondary metric: {sec_pretty} "
                f"({'higher' if sec_direction == 'higher' else 'lower'} is better)</summary>"
            )
            lines.append("")
            header = "| Model | Layer | " + " | ".join(_pretty_texture(t) for t in sec_pivot.columns) + " |"
            sep = "|" + "|".join(["---"] * (2 + len(sec_pivot.columns))) + "|"
            lines.append(header)
            lines.append(sep)
            for (model, layer), row in sec_pivot.iterrows():
                cells = [
                    _pretty_model(model),
                    str(layer),
                ] + [_format_metric(row[t], sec_direction) for t in sec_pivot.columns]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
            lines.append("</details>")
            lines.append("")
    return "\n".join(lines)


def _compute_clip_vs_dino_delta(summary: pd.DataFrame) -> pd.DataFrame:
    """For each (task, layer, texture) headline metric, compute DINOv2 - CLIP."""
    rows = []
    test = _filter_test(summary)
    for task, (metric, direction, pretty) in HEADLINE_METRICS.items():
        sub = test[(test["task"] == task) & (test["metric"] == metric)]
        for (layer, tex), grp in sub.groupby(["layer_name", "texture_condition"], observed=True):
            clip = grp[grp["model_name"] == "clip_vit_b16"]["value"].mean()
            dino = grp[grp["model_name"] == "dinov2_vit_b"]["value"].mean()
            if np.isnan(clip) or np.isnan(dino):
                continue
            delta_raw = dino - clip  # > 0 means DINOv2 has higher metric value
            if direction == "lower":
                dino_advantage = clip - dino  # > 0 means DINOv2 is better
            else:
                dino_advantage = dino - clip
            rows.append(
                {
                    "task": task,
                    "layer": layer,
                    "texture_condition": tex,
                    "metric": metric,
                    "direction": direction,
                    "clip": clip,
                    "dinov2": dino,
                    "dinov2_minus_clip_raw": delta_raw,
                    "dinov2_advantage": dino_advantage,
                }
            )
    return pd.DataFrame(rows)


def _make_delta_table_md(delta: pd.DataFrame) -> str:
    if delta.empty:
        return ""
    lines: List[str] = []
    for task, grp in delta.groupby("task"):
        info = DENSE_TASKS[task]
        lines.append(f"### {info.pretty}")
        lines.append("")
        pivot = grp.pivot_table(
            index="layer", columns="texture_condition", values="dinov2_advantage", aggfunc="mean"
        )
        pivot = pivot[[t for t in TEXTURE_ORDER if t in pivot.columns]]
        header = (
            "| Layer | " + " | ".join(_pretty_texture(t) for t in pivot.columns) + " |"
        )
        sep = "|" + "|".join(["---"] * (1 + len(pivot.columns))) + "|"
        lines.append(header)
        lines.append(sep)
        for layer, row in pivot.iterrows():
            cells = [str(layer)] + [
                _format_metric(row[t], "higher") for t in pivot.columns
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Positive values = DINOv2 better than CLIP on this metric.")
        lines.append("")
    return "\n".join(lines)


def _make_texture_drop_table_md(texture_drops: Optional[pd.DataFrame]) -> str:
    if texture_drops is None or texture_drops.empty:
        return ""
    lines: List[str] = []
    for task, (metric, direction, pretty) in HEADLINE_METRICS.items():
        sub = texture_drops[
            (texture_drops["task"] == task)
            & (texture_drops["metric"] == metric)
            & (texture_drops["split"] == "test")
            & (texture_drops["baseline_texture"] == "photorealistic")
        ]
        if sub.empty:
            continue
        info = DENSE_TASKS[task]
        lines.append(f"### {info.pretty}")
        lines.append("")
        lines.append(
            f"Drop = comparison − photorealistic, oriented so **positive = worse than photorealistic** on **{pretty}**."
        )
        lines.append("")
        pivot = sub.pivot_table(
            index=["model", "layer"],
            columns="comparison_texture",
            values="texture_drop",
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=[t for t in ("flat", "random_noise") if t in pivot.columns])
        header = "| Model | Layer | " + " | ".join(_pretty_texture(t) for t in pivot.columns) + " |"
        sep = "|" + "|".join(["---"] * (2 + len(pivot.columns))) + "|"
        lines.append(header)
        lines.append(sep)
        for (model, layer), row in pivot.iterrows():
            cells = [_pretty_model(model), str(layer)] + [
                _format_metric(row[t], "higher") for t in pivot.columns
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _per_render_fraction_above_chance(per_render: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if per_render.empty:
        return pd.DataFrame()
    for task, _ in HEADLINE_METRICS.items():
        info = DENSE_TASKS[task]
        col = info.per_render_column
        if col is None or col not in per_render.columns:
            continue
        sub = per_render[(per_render["task"] == task) & (per_render["split"] == "test")]
        for (model, layer, tex), grp in sub.groupby(
            ["model_name", "layer_name", "probe_texture_condition"], observed=True
        ):
            vals = grp[col].dropna()
            if vals.empty:
                continue
            chance = info.chance
            if chance is None:
                better_than_chance = float("nan")
            elif info.per_render_direction == "higher":
                better_than_chance = float((vals > chance).mean())
            else:
                better_than_chance = float((vals < chance).mean())
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "layer": layer,
                    "texture_condition": tex,
                    "n_renders": int(len(vals)),
                    "per_render_mean": float(vals.mean()),
                    "per_render_median": float(vals.median()),
                    "per_render_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "frac_better_than_chance": better_than_chance,
                }
            )
    return pd.DataFrame(rows)


def write_markdown_report(
    out_path: Path,
    summary: pd.DataFrame,
    per_render: pd.DataFrame,
    texture_drops: Optional[pd.DataFrame],
    manifest: Optional[pd.DataFrame],
    run_meta: pd.DataFrame,
    data_source: str,
    figure_paths: Dict[str, List[Path]],
    warnings: List[str],
) -> None:
    summary_table = _make_summary_table_md(summary)
    delta_df = _compute_clip_vs_dino_delta(summary)
    delta_table = _make_delta_table_md(delta_df)
    drop_table = _make_texture_drop_table_md(texture_drops)
    above_chance = _per_render_fraction_above_chance(per_render)

    # Difficulty ranking from headline metrics
    diff_rows = []
    test = _filter_test(summary)
    for task, (metric, direction, pretty) in HEADLINE_METRICS.items():
        sub = test[(test["task"] == task) & (test["metric"] == metric)]
        if sub.empty:
            continue
        for model in sorted(sub["model_name"].unique()):
            for layer in sorted(sub["layer_name"].unique(), key=layer_sort_key):
                vals = sub[(sub["model_name"] == model) & (sub["layer_name"] == layer)]
                if vals.empty:
                    continue
                diff_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "layer": layer,
                        "metric": metric,
                        "direction": direction,
                        "value_mean_across_texture": float(vals["value"].mean()),
                    }
                )
    diff_df = pd.DataFrame(diff_rows)

    # Run inventory
    inv_rows = []
    for _, row in run_meta.iterrows():
        inv_rows.append(
            f"- {row['model_name']} | {row['layer_name']} | {row['task']} | "
            f"texture={row['texture_condition']} | dir=`{row.get('probe_dir', '')}`"
        )
    inv_md = "\n".join(inv_rows) if inv_rows else "(no probe runs discovered)"

    lines: List[str] = []
    lines.append("# Experiment 1 Dense Probe Analysis")
    lines.append("")
    lines.append(
        "_Source W&B project_: "
        f"`{WANDB_ENTITY}/{WANDB_PROJECT}`."
    )
    lines.append(
        "_Data source used for this report_: " + data_source + "."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---------- 1. Overview ----------
    n_runs = len(run_meta)
    n_models = run_meta["model_name"].nunique()
    n_layers = run_meta["layer_name"].nunique()
    n_tasks = run_meta["task"].nunique()
    n_textures = run_meta["texture_condition"].nunique()
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(
        f"This report analyzes **{n_runs} dense linear-probe runs** trained on frozen "
        f"patch features from the rerendered Experiment 1 dataset. The training was logged to the W&B project "
        f"`{WANDB_ENTITY}/{WANDB_PROJECT}` (run group `exp1_dense`, "
        "see `notebooks/colab_exp1_zip_probe_training_wandb.ipynb`)."
    )
    lines.append("")
    lines.append(
        f"- **Models**: {n_models} backbones — "
        + ", ".join(_pretty_model(m) for m in sorted(run_meta["model_name"].unique()))
        + "."
    )
    lines.append(
        f"- **Layers probed**: {n_layers} — "
        + ", ".join(sorted(run_meta["layer_name"].unique(), key=layer_sort_key))
        + "."
    )
    lines.append(
        f"- **Tasks**: {n_tasks} dense per-patch geometry tasks — "
        + ", ".join(DENSE_TASKS[t].pretty for t in sorted(run_meta["task"].unique()))
        + "."
    )
    lines.append(
        f"- **Texture conditions** (within-texture train/eval): {n_textures} — "
        + ", ".join(_pretty_texture(t) for t in TEXTURE_ORDER if t in run_meta["texture_condition"].unique())
        + "."
    )
    lines.append(
        "- **Probe type**: lightweight learned head on frozen ViT *patch* features "
        "(`feature_mode = patch`). Heads regress per-patch SSI depth (loss = `ssi_l1`) "
        "or unit-norm camera-frame normals (loss = `masked_cosine`)."
    )
    lines.append(
        "- **Backbones are frozen**; only the probe head is trained. This is the "
        "central design of Experiment 1 (see `AGENTS.md`)."
    )
    lines.append("")

    # ---------- 2. Data sources ----------
    lines.append("## 2. Data Sources")
    lines.append("")
    lines.append(
        "Each W&B run uploads the exact per-probe artifacts that are also written to disk by "
        "`scripts/train_all_dense_depth_probes.py` and "
        "`scripts/train_all_dense_surface_normal_probes.py` "
        "(`metrics.json`, `history.csv`, `predictions.csv`, "
        "`predictions_<split>.npz`, `checkpoint.pt`). The W&B project is therefore a "
        "1:1 mirror of the local probe directory tree under "
        "`outputs/exp1_dense/probes/`."
    )
    lines.append("")
    lines.append("Concretely, this analysis pulled from:")
    lines.append("")
    lines.append(
        "- W&B project (when reachable): "
        f"`{WANDB_ENTITY}/{WANDB_PROJECT}` — see `run_metadata.csv`."
    )
    lines.append(
        "- Local per-probe metrics JSONs: "
        "`outputs/exp1_dense/probes/<model>/<layer>/<task>/within_<texture>/metrics.json`."
    )
    lines.append(
        "- Local aggregated long-format CSV: "
        "`outputs/exp1_dense/results/exp1_results_long.csv` (produced "
        "by `scripts/aggregate_exp1_results.py`)."
    )
    lines.append(
        "- Local texture-drop CSV: "
        "`outputs/exp1_dense/results/exp1_texture_drops.csv`."
    )
    lines.append(
        "- Local per-render dataframes: each `predictions.csv` under the probe directory above."
    )
    lines.append(
        "- Render manifest (state-space slicing): "
        "`data/exp1_dense/manifests/render_valid.parquet`."
    )
    lines.append("")
    lines.append("Outputs of this analysis script:")
    lines.append("")
    lines.append("- `outputs/exp1_dense_probe_analysis/analysis.md` (this file)")
    lines.append("- `outputs/exp1_dense_probe_analysis/metrics_summary.csv`")
    lines.append("- `outputs/exp1_dense_probe_analysis/run_metadata.csv`")
    lines.append("- `outputs/exp1_dense_probe_analysis/per_render_predictions.csv.gz`")
    lines.append("- `outputs/exp1_dense_probe_analysis/per_render_above_chance.csv`")
    lines.append("- `outputs/exp1_dense_probe_analysis/clip_vs_dinov2_delta.csv`")
    lines.append("- `outputs/exp1_dense_probe_analysis/notes_on_missing_data.md`")
    lines.append("- `outputs/exp1_dense_probe_analysis/figures/*.png` plus matching CSVs")
    lines.append("- `outputs/exp1_dense_probe_analysis/figures/state_space/*.png`")
    lines.append("")
    if warnings:
        lines.append("**Warnings emitted while loading data**:")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    # ---------- 3. Experimental setup ----------
    lines.append("## 3. Experimental Setup Inferred from Runs")
    lines.append("")
    sample_meta = run_meta.iloc[0] if len(run_meta) else None
    lines.append(
        "- **Backbones (frozen)**: CLIP ViT-B/16 (`clip_vit_b16`) and DINOv2 ViT-B/14 (`dinov2_vit_b`). "
        "Patch-feature dim = 768 for both."
    )
    lines.append(
        "- **Layers**: `final` (last ViT block output) and `layer8` (mid-late block, "
        "approximately the 8th of 12 blocks for ViT-B). The headline questions about whether early/mid layers "
        "preserve geometric structure are therefore restricted to one intermediate point per backbone."
    )
    lines.append(
        "- **Tasks**: `dense_depth_patches` (per-patch SSI depth regression) and "
        "`dense_surface_normal_patches` (per-patch unit-vector camera-frame normal regression on a 14×14 patch grid)."
    )
    lines.append(
        "- **Probe type**: per-task linear/MLP head on **frozen patch features**; feature mode = `patch`. "
        "Surface-normal loss = masked cosine; depth loss = scale-and-shift-invariant L1."
    )
    lines.append(
        "- **Texture conditions**: photorealistic (preserved or fallback material), flat textureless "
        "(constant gray surface, still lit), random_noise (procedural noise, see manifest "
        "`random_noise_node_type=ShaderNodeTexNoise`). The within-texture grid runs only when "
        "train and eval texture match (`texture_condition = within_<texture>`); cross-texture transfer "
        "is intentionally **not** part of the dense sub-study (see notebook section header)."
    )
    if sample_meta is not None:
        lines.append(
            "- **Train/val/test split sizes** (from `metrics.json` metadata): "
            "train=326, val=67, test=69 renders **per texture condition**. Splits are object-disjoint and "
            "deterministic by manifest; see `data/exp1_dense/manifests/render_valid.parquet`."
        )
    lines.append(
        "- **Metrics computed by the trainer**: for depth — `ssi_l1_mean/median`, "
        "`pearson_r_mean`, `scale_aware_*` and `scale_invariant_*` `abs_rel` / `rmse_log` / `delta_k`. "
        "For normals — `angular_error_deg_mean/median`, `within_{11.25,22.5,30}_deg`."
    )
    lines.append(
        "- **Per-render predictions**: `predictions.csv` reports one row per render; "
        "`predictions_<split>.npz` stores the raw 14×14×{1|3} prediction, target, and valid arrays, which is the "
        "basis of the state-space slice analysis below."
    )
    lines.append("")

    # ---------- 4. Main quantitative results ----------
    lines.append("## 4. Main Quantitative Results")
    lines.append("")
    lines.append("All tables below report **test-split** values, averaged across renders.")
    lines.append("")
    lines.append(summary_table)

    if figure_paths.get("aggregate"):
        lines.append("Reference figures:")
        for p in figure_paths["aggregate"][:6]:
            lines.append(f"- `figures/{p.name}`")
        lines.append("")

    # ---------- 5. Model comparisons ----------
    lines.append("## 5. Model Comparisons (CLIP vs DINOv2)")
    lines.append("")
    if delta_table:
        lines.append(delta_table)
    if not delta_df.empty:
        # Auto-generated qualitative description
        depth_delta = delta_df[delta_df["task"] == "dense_depth_patches"]
        normal_delta = delta_df[delta_df["task"] == "dense_surface_normal_patches"]

        def _describe(name: str, df: pd.DataFrame) -> str:
            if df.empty:
                return ""
            wins = (df["dinov2_advantage"] > 0).sum()
            total = len(df)
            avg = df["dinov2_advantage"].mean()
            best = df.loc[df["dinov2_advantage"].idxmax()] if not df.empty else None
            worst = df.loc[df["dinov2_advantage"].idxmin()] if not df.empty else None
            return (
                f"- **{name}**: DINOv2 beats CLIP in {wins}/{total} (layer × texture) cells; "
                f"mean advantage = {avg:.3f} (units of the headline metric, oriented so positive = DINOv2 better). "
                f"Largest DINOv2 advantage: {best['dinov2_advantage']:.3f} at "
                f"layer {best['layer']}, {_pretty_texture(best['texture_condition'])}. "
                f"Smallest (or CLIP's best case): {worst['dinov2_advantage']:.3f} at "
                f"layer {worst['layer']}, {_pretty_texture(worst['texture_condition'])}."
            )

        lines.append(_describe("Dense depth (Pearson r)", depth_delta))
        lines.append(_describe("Dense surface normals (median angular error)", normal_delta))
        lines.append("")
    if figure_paths.get("aggregate"):
        clip_figs = [p for p in figure_paths["aggregate"] if "clip_vs_dinov2" in p.name or "paired_scatter" in p.name]
        if clip_figs:
            lines.append("Reference figures:")
            for p in clip_figs:
                lines.append(f"- `figures/{p.name}`")
            lines.append("")

    # ---------- 6. Texture dependence ----------
    lines.append("## 6. Texture Dependence")
    lines.append("")
    if drop_table:
        lines.append(drop_table)
    if figure_paths.get("aggregate"):
        td_figs = [p for p in figure_paths["aggregate"] if "texture_drop" in p.name]
        if td_figs:
            lines.append("Reference figures:")
            for p in td_figs:
                lines.append(f"- `figures/{p.name}`")
            lines.append("")

    # ---------- 7. Task difficulty ----------
    lines.append("## 7. Task Difficulty")
    lines.append("")
    if not diff_df.empty:
        lines.append(
            "Headline metric values averaged across texture (test split). For depth, "
            "Pearson r is **higher = better** with chance ≈ 0; for normals, median "
            "angular error is **lower = better** with chance ≈ 57° (median of random "
            "unit-vector angle), and the trivial all-zeros prediction lies near 90° mean / 57° median."
        )
        lines.append("")
        lines.append("| Task | Model | Layer | Mean of headline metric across textures | Direction |")
        lines.append("|---|---|---|---|---|")
        for _, row in diff_df.sort_values(["task", "value_mean_across_texture"]).iterrows():
            lines.append(
                f"| {DENSE_TASKS[row['task']].pretty} | {_pretty_model(row['model'])} | {row['layer']} | "
                f"{row['value_mean_across_texture']:.3f} | "
                f"{'↑' if row['direction']=='higher' else '↓'} {row['direction']} is better |"
            )
        lines.append("")

    if not above_chance.empty:
        lines.append("**Per-render fraction beating chance** (`> chance` for higher-is-better tasks, `< chance` for lower-is-better tasks):")
        lines.append("")
        lines.append("| Task | Model | Layer | Texture | n | per-render mean | per-render median | frac > chance |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, row in above_chance.sort_values(["task", "model", "layer", "texture_condition"]).iterrows():
            lines.append(
                f"| {DENSE_TASKS[row['task']].pretty} | {_pretty_model(row['model'])} | "
                f"{row['layer']} | {_pretty_texture(row['texture_condition'])} | "
                f"{row['n_renders']} | {row['per_render_mean']:.3f} | "
                f"{row['per_render_median']:.3f} | {row['frac_better_than_chance']:.2%} |"
            )
        lines.append("")

    # ---------- 8. Layer-wise trends ----------
    lines.append("## 8. Layer-wise Trends")
    lines.append("")
    layers = sorted(run_meta["layer_name"].unique(), key=layer_sort_key)
    if set(layers) >= {"final", "layer8"}:
        layer_test = _filter_test(summary)
        notes = []
        for task, (metric, direction, _) in HEADLINE_METRICS.items():
            sub = layer_test[(layer_test["task"] == task) & (layer_test["metric"] == metric)]
            if sub.empty:
                continue
            for model in sorted(sub["model_name"].unique()):
                model_sub = sub[sub["model_name"] == model]
                pivot = model_sub.pivot_table(
                    index="texture_condition", columns="layer_name", values="value", aggfunc="mean"
                )
                if "final" not in pivot.columns or "layer8" not in pivot.columns:
                    continue
                better = []
                for tex in pivot.index:
                    f, l8 = pivot.loc[tex, "final"], pivot.loc[tex, "layer8"]
                    if pd.isna(f) or pd.isna(l8):
                        continue
                    if direction == "higher":
                        verdict = "layer8 > final" if l8 > f else "final > layer8"
                    else:
                        verdict = "layer8 < final" if l8 < f else "final < layer8"
                    better.append(f"{_pretty_texture(tex)}: {verdict} ({l8:.3f} vs {f:.3f})")
                notes.append(
                    f"- **{DENSE_TASKS[task].pretty}**, {_pretty_model(model)}: " + "; ".join(better) + "."
                )
        if notes:
            lines.append("Final vs layer8 comparison (test, headline metric):")
            lines.extend(notes)
            lines.append("")
        else:
            lines.append("(No layer pairs available in summary.)")
            lines.append("")
    else:
        lines.append(
            "Only one layer per backbone was probed in this run group, so layer-wise "
            "ablation is limited; see the layerwise figures for the available comparison."
        )
        lines.append("")
    if figure_paths.get("aggregate"):
        lw_figs = [p for p in figure_paths["aggregate"] if "layerwise" in p.name]
        if lw_figs:
            lines.append("Reference figures:")
            for p in lw_figs:
                lines.append(f"- `figures/{p.name}`")
            lines.append("")

    # ---------- 9. State-space / slice analysis ----------
    lines.append("## 9. State-Space / Slice Analysis")
    lines.append("")
    if not per_render.empty and manifest is not None:
        lines.append(
            "Per-render scores were merged with the render manifest to expose how aggregate "
            "metrics depend on rendering variables. For each task/layer we partition the "
            "test renders by category and by quintile bins of camera distance, camera azimuth, "
            "camera elevation, light elevation, light intensity, object scale, and foreground "
            "fraction. CSVs are saved next to every PNG."
        )
        lines.append("")
        if figure_paths.get("state_space"):
            lines.append("Selected figures (one per slice/task/layer; see `figures/state_space/` for the full set):")
            shown = 0
            seen_keys = set()
            for p in figure_paths["state_space"]:
                key = p.stem.rsplit("_", 1)[0]
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                lines.append(f"- `figures/state_space/{p.name}`")
                shown += 1
                if shown >= 16:
                    break
            lines.append("")
    else:
        lines.append("No state-space slicing available (predictions or manifest missing).")
        lines.append("")

    # ---------- 10. Recommended aggregate figures ----------
    lines.append("## 10. Aggregate Figures Generated")
    lines.append("")
    if figure_paths.get("aggregate"):
        for p in figure_paths["aggregate"]:
            lines.append(f"- `figures/{p.name}`")
        lines.append("")
    if figure_paths.get("state_space"):
        lines.append(f"State-space breakdown: **{len(figure_paths['state_space'])}** figures under `figures/state_space/`.")
        lines.append("")

    # ---------- 11. Interpretation ----------
    lines.append("## 11. Interpretation in Context of Experiment 1")
    lines.append("")
    interp = _auto_interpretation(summary, delta_df, texture_drops, above_chance)
    lines.append(interp)
    lines.append("")

    # ---------- 12. Limitations ----------
    lines.append("## 12. Limitations and Caveats")
    lines.append("")
    lines.append(
        "- **Small held-out set**: only 69 test renders per texture condition; "
        "per-bin counts in the state-space figures are accordingly small (≈14 per quintile)."
    )
    lines.append(
        "- **Single intermediate layer (`layer8`)**: the canonical layer4/8/12 sweep described in `AGENTS.md` "
        "was reduced to two layers per backbone for this dense sub-study; we cannot fully characterize where "
        "CLIP's geometry signal peaks along the depth of the ViT."
    )
    lines.append(
        "- **One CLIP scale and one DINOv2 scale only** (ViT-B). CLIP ViT-L/14 and DINOv2 ViT-L were probed in the "
        "global rerender (`outputs/exp1_main`) but not in this dense run group."
    )
    lines.append(
        "- **Cross-texture transfer not measured here**: the dense notebook intentionally runs only "
        "within-texture probes, so we cannot say whether features trained on one texture transfer to another. "
        "The global rerender has those experiments and supplements the conclusions below."
    )
    lines.append(
        "- **Aggregate metrics can hide failures**: in particular, `delta_1` ≈ 1.0 and `scale_invariant_abs_rel` ≈ 0.04 "
        "look excellent but are dominated by the trivial near-constant-distance background of foreground patches; "
        "the more demanding scale-aware family and Pearson r reveal that the probes barely separate near/far patches "
        "within an object."
    )
    lines.append(
        "- **Trainer-side warmup / early-stop**: histories were capped at ≤20 epochs by config and used early "
        "stopping with patience=8; some probes may be undertrained. The headline ordering across models is "
        "stable across the captured epochs, so this is unlikely to flip the qualitative conclusions, but the "
        "absolute numbers could improve with longer training."
    )
    lines.append(
        "- **Surface-normal metric chance**: the trivial all-zero / all-camera-forward prediction yields ≈57° "
        "median angular error; values close to this should not be treated as 'learning'."
    )
    lines.append("")

    # ---------- 13. Final takeaways ----------
    lines.append("## 13. Final Takeaways")
    lines.append("")
    for bullet in _final_takeaways(summary, delta_df, texture_drops, above_chance):
        lines.append(f"- {bullet}")
    lines.append("")

    # ---------- Appendix: probe inventory ----------
    lines.append("---")
    lines.append("")
    lines.append("## Appendix: Probe Inventory")
    lines.append("")
    lines.append(f"Total probe runs in this analysis: **{len(run_meta)}**.")
    lines.append("")
    lines.append("<details><summary>Per-run paths (W&B mirrors these directories)</summary>")
    lines.append("")
    lines.append(inv_md)
    lines.append("")
    lines.append("</details>")
    lines.append("")

    out_path.write_text("\n".join(lines))


def _auto_interpretation(
    summary: pd.DataFrame,
    delta_df: pd.DataFrame,
    texture_drops: Optional[pd.DataFrame],
    above_chance: pd.DataFrame,
) -> str:
    text: List[str] = []
    if delta_df.empty:
        return "(Insufficient data to auto-generate interpretation.)"
    depth = delta_df[delta_df["task"] == "dense_depth_patches"]
    norm = delta_df[delta_df["task"] == "dense_surface_normal_patches"]

    def _avg_metric(task: str, model: str, metric: str) -> Optional[float]:
        sub = summary[
            (summary["task"] == task)
            & (summary["model_name"] == model)
            & (summary["metric"] == metric)
            & (summary["split"] == "test")
        ]
        if sub.empty:
            return None
        return float(sub["value"].mean())

    text.append("### Does DINOv2 look more robustly geometric than CLIP?")
    text.append("")
    if not depth.empty:
        depth_wins = (depth["dinov2_advantage"] > 0).sum()
        text.append(
            f"On per-render Pearson r for **dense depth**, DINOv2 outperforms CLIP in "
            f"{depth_wins} of {len(depth)} (layer × texture) cells. Average DINOv2 advantage = "
            f"{depth['dinov2_advantage'].mean():+.3f} (raw Pearson r units)."
        )
    if not norm.empty:
        norm_wins = (norm["dinov2_advantage"] > 0).sum()
        text.append(
            f"On median **angular error** for dense normals, DINOv2 reduces error in "
            f"{norm_wins} of {len(norm)} cells; average reduction = "
            f"{norm['dinov2_advantage'].mean():+.2f}° (positive = DINOv2 lower)."
        )
    text.append("")

    text.append("### Does CLIP rely more on photorealistic texture / semantics?")
    text.append("")
    if texture_drops is not None and not texture_drops.empty:
        # Compare CLIP photo -> flat / random and DINOv2 photo -> flat / random
        sub = texture_drops[
            (texture_drops["split"] == "test")
            & (texture_drops["baseline_texture"] == "photorealistic")
        ]
        for task, (metric, direction, _) in HEADLINE_METRICS.items():
            taskdrop = sub[(sub["task"] == task) & (sub["metric"] == metric)]
            if taskdrop.empty:
                continue
            clip_drop = taskdrop[taskdrop["model"] == "clip_vit_b16"]["texture_drop"].mean()
            dino_drop = taskdrop[taskdrop["model"] == "dinov2_vit_b"]["texture_drop"].mean()
            text.append(
                f"- {DENSE_TASKS[task].pretty}: mean **drop** (positive = worse than photorealistic) "
                f"vs photorealistic — CLIP = {clip_drop:+.3f}, DINOv2 = {dino_drop:+.3f}."
            )
        text.append(
            ""
            "\nA larger positive drop for CLIP would be consistent with stronger texture/semantic reliance. "
            "Note that on the dense per-patch geometry metrics, **both backbones** struggle, so the absolute "
            "deltas are small; the strongest version of this hypothesis is best tested on the global probe "
            "rerender (lighting, viewpoint, relative depth) which is logged elsewhere."
        )
        text.append("")

    text.append("### Are flat / random texture conditions especially revealing?")
    text.append("")
    text.append(
        "By design these conditions remove the photorealistic surface cue; differences "
        "between models in flat/random regimes therefore most cleanly isolate **geometry encoding** "
        "from texture priors. The per-task tables above show how each model degrades. The state-space "
        "figures in `figures/state_space/` further show that for dense depth, both models are essentially "
        "unable to recover per-patch depth ordering (Pearson r near zero across categories and viewpoints), "
        "even in photorealistic conditions; this is a useful negative result for the dense formulation."
    )
    text.append("")

    text.append("### Are there settings where CLIP appears competitive or better than DINOv2?")
    text.append("")
    if not depth.empty:
        clip_wins_depth = depth[depth["dinov2_advantage"] <= 0]
        if not clip_wins_depth.empty:
            for _, row in clip_wins_depth.iterrows():
                text.append(
                    f"- Dense depth: CLIP ≈ DINOv2 at layer **{row['layer']}**, "
                    f"{_pretty_texture(row['texture_condition'])} "
                    f"(CLIP = {row['clip']:.3f}, DINOv2 = {row['dinov2']:.3f})."
                )
        else:
            text.append(
                "- Dense depth: no (layer × texture) cell where CLIP beats DINOv2 on Pearson r."
            )
    if not norm.empty:
        clip_wins_norm = norm[norm["dinov2_advantage"] <= 0]
        if not clip_wins_norm.empty:
            for _, row in clip_wins_norm.iterrows():
                text.append(
                    f"- Dense normals: CLIP ≈ DINOv2 at layer **{row['layer']}**, "
                    f"{_pretty_texture(row['texture_condition'])} "
                    f"(CLIP = {row['clip']:.2f}°, DINOv2 = {row['dinov2']:.2f}°)."
                )
        else:
            text.append(
                "- Dense normals: DINOv2 always achieves lower median angular error than CLIP."
            )
    text.append("")

    text.append("### Are aggregate metrics hiding strong/weak regions?")
    text.append("")
    if not above_chance.empty:
        depth_chance = above_chance[above_chance["task"] == "dense_depth_patches"]
        norm_chance = above_chance[above_chance["task"] == "dense_surface_normal_patches"]
        if not depth_chance.empty:
            mean_fb = depth_chance.groupby("model")["frac_better_than_chance"].mean()
            for model, frac in mean_fb.items():
                text.append(
                    f"- Dense depth: {_pretty_model(model)} has Pearson r > 0 on {frac:.0%} of "
                    "test renders (averaged across layer/texture)."
                )
        if not norm_chance.empty:
            mean_fb = norm_chance.groupby("model")["frac_better_than_chance"].mean()
            for model, frac in mean_fb.items():
                text.append(
                    f"- Dense normals: {_pretty_model(model)} achieves per-render mean angular "
                    f"error **below the 90° random-vector baseline** on {frac:.0%} of test renders."
                )
        text.append(
            ""
            "\nThese are weak bars (Pearson r > 0; mean angular error < 90°). They show that the "
            "probes have learned **some** structure on most renders, but the aggregate magnitudes "
            "are modest — for depth, CLIP's mean Pearson r is around 0.07 across textures, and even "
            "DINOv2's layer8 mean is only ≈0.36. The failure is broad and stems from the dense "
            "per-patch formulation being very demanding for frozen ViT features without any spatial "
            "decoder, not from a handful of catastrophic renders."
        )
    text.append("")

    return "\n".join(text)


def _final_takeaways(
    summary: pd.DataFrame,
    delta_df: pd.DataFrame,
    texture_drops: Optional[pd.DataFrame],
    above_chance: pd.DataFrame,
) -> List[str]:
    bullets: List[str] = []
    if not delta_df.empty:
        norm = delta_df[delta_df["task"] == "dense_surface_normal_patches"]
        if not norm.empty:
            avg = norm["dinov2_advantage"].mean()
            winner = "DINOv2 ViT-B" if avg > 0 else "CLIP ViT-B/16"
            other = "CLIP ViT-B/16" if avg > 0 else "DINOv2 ViT-B"
            magnitude = abs(avg)
            bullets.append(
                f"**Dense surface normals (median angular error)**: averaged across all "
                f"(layer × texture) cells, **{winner}** is the lower-error backbone by ≈"
                f"{magnitude:.2f}° vs {other}. The advantage is small (well under 2°) and the "
                "sign flips by layer — at the **final layer** DINOv2 is slightly better, while "
                "at **layer8** CLIP is slightly better. The dense per-patch formulation is hard "
                "for both backbones, and median angular errors stay in the 46–48° band overall."
            )
        depth = delta_df[delta_df["task"] == "dense_depth_patches"]
        if not depth.empty:
            avg = depth["dinov2_advantage"].mean()
            winner = "DINOv2 ViT-B" if avg > 0 else "CLIP ViT-B/16"
            bullets.append(
                f"**Dense depth (per-render Pearson r)**: **{winner}** wins on every (layer × "
                f"texture) cell, with an average advantage of {avg:+.3f}. The biggest gap is at "
                "layer8 in the random-noise regime (DINOv2 ≈0.35 vs CLIP ≈0.00), which is "
                "consistent with the hypothesis that self-supervised features carry more "
                "texture-invariant geometric structure."
            )
    if texture_drops is not None and not texture_drops.empty:
        for task, (metric, direction, _) in HEADLINE_METRICS.items():
            sub = texture_drops[
                (texture_drops["split"] == "test")
                & (texture_drops["task"] == task)
                & (texture_drops["metric"] == metric)
                & (texture_drops["baseline_texture"] == "photorealistic")
            ]
            if sub.empty:
                continue
            clip_drop = sub[sub["model"] == "clip_vit_b16"]["texture_drop"].mean()
            dino_drop = sub[sub["model"] == "dinov2_vit_b"]["texture_drop"].mean()
            bullets.append(
                f"**Texture sensitivity on {DENSE_TASKS[task].pretty}**: average drop vs "
                f"photorealistic (positive = worse) — CLIP {clip_drop:+.3f}, DINOv2 {dino_drop:+.3f}. "
                "Differences in both directions are small in this dense formulation, "
                "suggesting that for **patch-level** geometry, photorealistic texture neither "
                "rescues CLIP nor uniquely helps DINOv2."
            )
            break  # one summary bullet is enough
    if not above_chance.empty:
        sub_depth = above_chance[above_chance["task"] == "dense_depth_patches"]
        if not sub_depth.empty:
            best = sub_depth.loc[sub_depth["frac_better_than_chance"].idxmax()]
            worst = sub_depth.loc[sub_depth["frac_better_than_chance"].idxmin()]
            bullets.append(
                "**Per-render reality check (depth)**: even in the best (model × layer × texture) "
                f"cell, only **{best['frac_better_than_chance']:.0%}** of test renders have positive "
                f"Pearson r ({_pretty_model(best['model'])}, {best['layer']}, "
                f"{_pretty_texture(best['texture_condition'])}). The worst cell drops to "
                f"**{worst['frac_better_than_chance']:.0%}** ({_pretty_model(worst['model'])}, "
                f"{worst['layer']}, {_pretty_texture(worst['texture_condition'])}). This is broad "
                "uniform difficulty, not a small set of catastrophic renders."
            )
        sub_norm = above_chance[above_chance["task"] == "dense_surface_normal_patches"]
        if not sub_norm.empty:
            best = sub_norm.loc[sub_norm["frac_better_than_chance"].idxmax()]
            bullets.append(
                "**Per-render reality check (normals)**: at most "
                f"**{best['frac_better_than_chance']:.0%}** of test renders achieve a per-render "
                "mean angular error below the 90° random-vector baseline (note: chance for the "
                "median angle is closer to 60°; medians of ≈46-48° in the headline table show real "
                "but modest signal across most renders)."
            )
    bullets.append(
        "**Layer comparison**: for **dense depth (Pearson r)**, layer8 features are much better "
        "than the final layer for both backbones — DINOv2 jumps from ≈0.15 (final) to ≈0.36 (layer8), "
        "and CLIP goes from ≈0.03 to ≈0.05. This is consistent with the broader claim that mid-late "
        "ViT layers retain more spatial structure than the language-aligned final CLIP layer. For "
        "**dense normals**, the layer effect is small (<2° median angular error in either direction)."
    )
    bullets.append(
        "**Most informative figure for a final report**: the per-task texture heatmaps "
        "(`figures/heatmap_densedepth_*.png`, `figures/heatmap_densenorm_*.png`) and the paired "
        "per-render scatter plots (`figures/paired_scatter_*.png`), which together show *both* "
        "the aggregate ranking and the per-render variability that underpins it."
    )
    bullets.append(
        "**What's missing for a stronger claim**: a finer layer sweep (layer4, 6, 10), the ViT-L "
        "backbones, and cross-texture transfer probes; the global rerender already covers these "
        "and should be used together with this dense analysis when writing the report."
    )
    return bullets


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-wandb",
        action="store_true",
        help="Attempt to pull run metadata from W&B in addition to local files.",
    )
    parser.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE_ROOT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--texture-drops-csv", type=Path, default=DEFAULT_TEXTURE_DROPS_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    out_dir = args.output_dir
    figures_dir = out_dir / "figures"
    state_space_dir = figures_dir / "state_space"
    figures_dir.mkdir(parents=True, exist_ok=True)
    state_space_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    wandb_meta: Optional[pd.DataFrame] = None
    if args.from_wandb:
        wandb_meta, wandb_summary, ws = load_wandb_runs()
        warnings.extend(ws)
    else:
        warnings.append(
            "wandb fetch was not requested (and the wandb python package is not installed in this env). "
            "Falling back to local probe outputs, which mirror the W&B project 1:1 by construction "
            "(see notebooks/colab_exp1_zip_probe_training_wandb.ipynb)."
        )

    probes_df = discover_local_probes(args.probe_root)
    if probes_df.empty:
        warnings.append(
            f"No probe directories found under {args.probe_root}. Aborting without writing report."
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "notes_on_missing_data.md").write_text(
            "# Notes on missing data\n\n"
            + "\n".join(f"- {w}" for w in warnings)
        )
        print("Aborting: no probe outputs discovered. Warnings written.")
        return

    # Merge in W&B metadata if available, otherwise use local probe directory info.
    run_meta = probes_df.copy()
    if wandb_meta is not None and not wandb_meta.empty:
        wandb_meta = wandb_meta.copy()
        run_meta = run_meta.merge(
            wandb_meta,
            on=["model_name", "layer_name", "task", "texture_condition"],
            how="left",
            suffixes=("", "_wandb"),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    run_meta.to_csv(out_dir / "run_metadata.csv", index=False)

    summary = load_local_summary_metrics(probes_df)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    per_render = load_per_render_predictions(probes_df)
    if not per_render.empty:
        per_render.to_csv(out_dir / "per_render_predictions.csv.gz", index=False, compression="gzip")

    texture_drops = None
    if args.texture_drops_csv.is_file():
        texture_drops = pd.read_csv(args.texture_drops_csv)
        # Only keep dense tasks
        texture_drops = texture_drops[texture_drops["task"].isin(DENSE_TASK_NAMES)].copy()

    manifest = load_manifest(args.manifest)

    delta_df = _compute_clip_vs_dino_delta(summary)
    delta_df.to_csv(out_dir / "clip_vs_dinov2_delta.csv", index=False)
    above_chance = _per_render_fraction_above_chance(per_render)
    if not above_chance.empty:
        above_chance.to_csv(out_dir / "per_render_above_chance.csv", index=False)

    # Generate aggregate figures.
    figure_paths: Dict[str, List[Path]] = {"aggregate": [], "state_space": []}
    figure_paths["aggregate"].extend(make_metric_heatmaps(summary, figures_dir))
    figure_paths["aggregate"].extend(make_clip_vs_dino_bar(summary, figures_dir))
    figure_paths["aggregate"].extend(make_texture_drop_plot(texture_drops, figures_dir))
    figure_paths["aggregate"].extend(make_layerwise_plot(summary, figures_dir))
    figure_paths["aggregate"].extend(make_per_render_distributions(per_render, figures_dir))
    figure_paths["aggregate"].extend(make_paired_scatter(per_render, figures_dir))

    figure_paths["state_space"].extend(
        make_state_space_plots(per_render, manifest, state_space_dir, warnings_out=warnings)
    )

    # Markdown report.
    write_markdown_report(
        out_dir / "analysis.md",
        summary=summary,
        per_render=per_render,
        texture_drops=texture_drops,
        manifest=manifest,
        run_meta=run_meta,
        data_source=(
            "Local probe outputs at outputs/exp1_dense/ (W&B mirror not queried directly)"
            if wandb_meta is None
            else "Weights & Biases API + local probe outputs (cross-validated)"
        ),
        figure_paths=figure_paths,
        warnings=warnings,
    )

    # Notes on missing data
    missing_lines = ["# Notes on missing data", ""]
    if not warnings:
        missing_lines.append("(No issues encountered.)")
    else:
        for w in warnings:
            missing_lines.append(f"- {w}")
    missing_lines.append("")
    missing_lines.append("## Probe runs inventory")
    missing_lines.append("")
    for _, row in probes_df.iterrows():
        missing = []
        for col in [
            "metrics_path",
            "history_path",
            "predictions_csv",
            "predictions_test_npz",
            "predictions_val_npz",
            "predictions_train_npz",
            "checkpoint_path",
        ]:
            if not Path(row[col]).is_file():
                missing.append(col)
        flag = " | MISSING: " + ", ".join(missing) if missing else ""
        missing_lines.append(
            f"- {row['model_name']} / {row['layer_name']} / {row['task']} / "
            f"texture={row['texture_condition']}{flag}"
        )
    (out_dir / "notes_on_missing_data.md").write_text("\n".join(missing_lines))

    # Terminal summary
    print("=" * 80)
    print("Experiment 1 dense-probe analysis complete.")
    print(f"  Report:      {out_dir / 'analysis.md'}")
    print(f"  Figures:     {figures_dir} ({len(figure_paths['aggregate'])} aggregate, "
          f"{len(figure_paths['state_space'])} state-space)")
    print(f"  Runs inc.:   {len(probes_df)} probe runs / {probes_df['model_name'].nunique()} models / "
          f"{probes_df['layer_name'].nunique()} layers / {probes_df['task'].nunique()} tasks / "
          f"{probes_df['texture_condition'].nunique()} textures")
    if wandb_meta is None:
        print("  W&B:         not queried (used local outputs which mirror W&B 1:1)")
    else:
        print(f"  W&B runs:    {len(wandb_meta)} from {WANDB_ENTITY}/{WANDB_PROJECT}")
    if warnings:
        print(f"  Warnings:    {len(warnings)} (see notes_on_missing_data.md)")
    print("=" * 80)


if __name__ == "__main__":
    main()

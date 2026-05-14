#!/usr/bin/env python3
"""State-space breakdown of per-render probe performance.

For each global (main) probe task and each dense probe task, we:

1. Collect every render's per-render score from ``predictions.csv``.
2. Merge with the render manifest (camera distance, azimuth, elevation,
   lighting, object scale, category, ...).
3. Stratify performance by rendering variables and plot:
   - per-category box plots
   - per-camera-distance bin lines
   - per-azimuth bin lines (polar-ish plots)
   - per-elevation bin lines
   - per-light-elevation bin lines

Each figure is faceted by model and texture condition, and uses the per-task
'direction' (higher / lower is better) to keep the y-axis interpretation
consistent.

This script is the per-render variability counterpart to
``plot_exp1_rerender_summary.py`` and is designed to reveal whether a model's
aggregate metric is uniform across the state space or driven by a subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis._common import (  # noqa: E402
    ALL_TASKS,
    DENSE_TASKS,
    GLOBAL_TASKS,
    MODEL_COLORS,
    MODEL_DISPLAY,
    TEXTURE_COLORS,
    TEXTURE_DISPLAY,
    TEXTURE_ORDER,
    TaskInfo,
    get_pyplot,
    layer_sort_key,
    load_manifest,
    load_per_render_table,
    merge_with_manifest,
    save_dataframe,
    save_figure,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument(
        "--main-manifest",
        type=Path,
        default=Path("data/exp1_under12h/manifests/render_valid.parquet"),
    )
    parser.add_argument(
        "--dense-manifest",
        type=Path,
        default=Path("data/exp1_under12h_dense/manifests/render_valid.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/exp1_rerender_analysis_figures/state_space"),
    )
    parser.add_argument("--split", default="test")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def collect_per_render(probe_root: Path, family: str, tasks: Sequence[str], split: str) -> pd.DataFrame:
    df = load_per_render_table(probe_root, family=family, tasks=tasks)
    if df.empty:
        return df
    df = df[df["split"] == split].copy()
    # standardize a "score" column based on the task's per-render direction
    score_rows: List[pd.DataFrame] = []
    for task, info in ALL_TASKS.items():
        if task not in tasks:
            continue
        col = info.per_render_column
        if col is None:
            continue
        sub = df[df["task"] == task].copy()
        if sub.empty or col not in sub.columns:
            continue
        sub["score"] = sub[col].astype(float)
        sub["score_direction"] = info.per_render_direction
        sub["score_metric"] = col
        score_rows.append(sub)
    if not score_rows:
        return pd.DataFrame()
    return pd.concat(score_rows, ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _ordered_models(df: pd.DataFrame) -> List[str]:
    preferred = ["clip_vit_b16", "clip_vit_l14", "dinov2_vit_b", "dinov2_vit_l"]
    pres = [m for m in preferred if m in df["model"].unique()]
    pres.extend(sorted(m for m in df["model"].unique() if m not in pres))
    return pres


def _binned_lines(
    df: pd.DataFrame,
    x_col: str,
    bins: np.ndarray,
    out_dir: Path,
    title_prefix: str,
    filename: str,
    score_label: str,
    chance: Optional[float] = None,
) -> None:
    if df.empty or x_col not in df.columns:
        return
    # Ensure unique, ascending bin edges; otherwise this state-space variable
    # is effectively constant and we should not produce a misleading plot.
    bins = np.unique(np.asarray(bins, dtype=float))
    if len(bins) < 2:
        return
    plt = get_pyplot()
    models = _ordered_models(df)
    textures = [t for t in TEXTURE_ORDER if t in df["texture_condition"].unique()]
    fig, axes = plt.subplots(1, len(textures), figsize=(3.6 * len(textures) + 1.5, 4.0), sharey=True)
    if len(textures) == 1:
        axes = [axes]
    tab_rows: List[dict] = []
    for ax, tex in zip(axes, textures):
        for m in models:
            sub = df[(df["model"] == m) & (df["texture_condition"] == tex)].dropna(subset=[x_col, "score"])
            if sub.empty:
                continue
            cats = pd.cut(sub[x_col], bins=bins, include_lowest=True)
            grp = sub.groupby(cats, observed=True)["score"]
            mid = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
            mean = grp.mean()
            # align to bin centers
            mids_present = [m_ for m_, c_ in zip(mid, [i for i in mean.index]) if c_ in mean.index]
            mids_array = np.array([mid[i] for i, k in enumerate(mean.index)]) if len(mean) > 0 else np.array([])
            xs = mids_array
            ys = mean.values
            counts = grp.count().values
            ax.plot(
                xs,
                ys,
                marker="o",
                color=MODEL_COLORS.get(m, "k"),
                linewidth=2,
                label=MODEL_DISPLAY.get(m, m) if ax is axes[0] else None,
            )
            for x_, y_, c_ in zip(xs, ys, counts):
                tab_rows.append(
                    dict(
                        model=m,
                        texture_condition=tex,
                        x_col=x_col,
                        bin_center=x_,
                        score_mean=y_,
                        n=int(c_),
                    )
                )
        if chance is not None:
            ax.axhline(chance, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_title(TEXTURE_DISPLAY[tex])
        ax.grid(linestyle=":", alpha=0.5)
    axes[0].set_ylabel(score_label)
    fig.suptitle(title_prefix)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.05), fontsize=8
        )
    fig.tight_layout()
    save_figure(fig, out_dir / f"{filename}.png")
    save_dataframe(pd.DataFrame(tab_rows), out_dir / f"{filename}.csv")
    plt.close(fig)


def _category_box(
    df: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    filename: str,
    score_label: str,
    chance: Optional[float] = None,
) -> None:
    if df.empty or "category" not in df.columns:
        return
    cats = sorted(df["category"].dropna().unique())
    if not cats:
        return
    plt = get_pyplot()
    textures = [t for t in TEXTURE_ORDER if t in df["texture_condition"].unique()]
    models = _ordered_models(df)
    fig, axes = plt.subplots(len(textures), 1, figsize=(2.0 + 1.0 * len(cats), 2.8 * len(textures)), sharex=True)
    if len(textures) == 1:
        axes = [axes]
    tab_rows: List[dict] = []
    for ax, tex in zip(axes, textures):
        x_positions: List[float] = []
        labels: List[str] = []
        for ci, cat in enumerate(cats):
            for mi, model in enumerate(models):
                sub = df[(df["category"] == cat) & (df["model"] == model) & (df["texture_condition"] == tex)]
                vals = sub["score"].dropna().values
                if len(vals) == 0:
                    continue
                pos = ci * (len(models) + 1) + mi
                bp = ax.boxplot(
                    [vals],
                    positions=[pos],
                    widths=0.7,
                    patch_artist=True,
                    showfliers=False,
                )
                for box in bp["boxes"]:
                    box.set_facecolor(MODEL_COLORS.get(model, "lightgrey"))
                    box.set_alpha(0.7)
                jitter = np.random.RandomState(ci).uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(np.full_like(vals, pos) + jitter, vals, s=6, alpha=0.4, color="black")
                x_positions.append(pos)
                labels.append(f"{cat}\n{MODEL_DISPLAY.get(model, model)}")
                tab_rows.append(
                    dict(
                        category=cat,
                        model=model,
                        texture_condition=tex,
                        n=len(vals),
                        median=float(np.median(vals)),
                        mean=float(np.mean(vals)),
                    )
                )
        if chance is not None:
            ax.axhline(chance, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"{title_prefix} - {TEXTURE_DISPLAY[tex]}")
        ax.set_ylabel(score_label)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        # Only label categories, with model legend below
        cat_centers = [ci * (len(models) + 1) + (len(models) - 1) / 2 for ci in range(len(cats))]
        ax.set_xticks(cat_centers)
        ax.set_xticklabels(cats, rotation=15)
    # legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=MODEL_COLORS.get(m, "grey"), label=MODEL_DISPLAY.get(m, m), alpha=0.7)
        for m in models
    ]
    axes[0].legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / f"{filename}.png")
    save_dataframe(pd.DataFrame(tab_rows), out_dir / f"{filename}.csv")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-task panel
# ---------------------------------------------------------------------------


def state_space_for_task(
    df_all: pd.DataFrame,
    task: str,
    layer: str,
    out_dir: Path,
) -> None:
    info = ALL_TASKS[task]
    df = df_all[(df_all["task"] == task) & (df_all["layer"] == layer)].copy()
    if df.empty:
        return
    base = out_dir / f"{info.short}_{layer}"
    base.mkdir(parents=True, exist_ok=True)

    score_label = f"per-render {info.per_render_column} ({info.per_render_direction} = better)"
    chance = info.chance if info.per_render_direction in ("higher", "lower") else None

    # Category boxplot
    _category_box(
        df,
        base,
        title_prefix=f"{info.pretty} (layer={layer})",
        filename="by_category",
        score_label=score_label,
        chance=chance,
    )
    # Camera distance bins (only if not effectively constant)
    if "camera_distance" in df.columns and df["camera_distance"].notna().any():
        lo, hi = df["camera_distance"].quantile([0.02, 0.98])
        if hi - lo > 1e-6:
            bins = np.linspace(float(lo), float(hi), 6)
            _binned_lines(df, "camera_distance", bins, base, f"{info.pretty} (layer={layer}) - by camera distance", "by_camera_distance", score_label, chance)
    # Azimuth
    if "camera_azimuth_deg" in df.columns and df["camera_azimuth_deg"].notna().any():
        bins = np.linspace(-180, 180, 13)
        _binned_lines(df, "camera_azimuth_deg", bins, base, f"{info.pretty} (layer={layer}) - by camera azimuth", "by_camera_azimuth", score_label, chance)
    # Elevation
    if "camera_elevation_deg" in df.columns and df["camera_elevation_deg"].notna().any():
        lo, hi = df["camera_elevation_deg"].min() - 1, df["camera_elevation_deg"].max() + 1
        if hi - lo > 1e-6:
            bins = np.linspace(lo, hi, 6)
            _binned_lines(df, "camera_elevation_deg", bins, base, f"{info.pretty} (layer={layer}) - by camera elevation", "by_camera_elevation", score_label, chance)
    # Light elevation
    if "light_elevation_deg" in df.columns and df["light_elevation_deg"].notna().any():
        lo, hi = df["light_elevation_deg"].min() - 1, df["light_elevation_deg"].max() + 1
        if hi - lo > 1e-6:
            bins = np.linspace(lo, hi, 6)
            _binned_lines(df, "light_elevation_deg", bins, base, f"{info.pretty} (layer={layer}) - by light elevation", "by_light_elevation", score_label, chance)
    # Object scale
    if "object_scale" in df.columns and df["object_scale"].notna().any():
        lo, hi = df["object_scale"].quantile([0.02, 0.98])
        if hi > lo:
            bins = np.linspace(float(lo), float(hi), 6)
            _binned_lines(df, "object_scale", bins, base, f"{info.pretty} (layer={layer}) - by object scale", "by_object_scale", score_label, chance)
    # Foreground fraction
    if "qc_foreground_fraction" in df.columns and df["qc_foreground_fraction"].notna().any():
        lo, hi = df["qc_foreground_fraction"].quantile([0.02, 0.98])
        if hi > lo:
            bins = np.linspace(float(lo), float(hi), 6)
            _binned_lines(df, "qc_foreground_fraction", bins, base, f"{info.pretty} (layer={layer}) - by foreground fraction", "by_foreground_fraction", score_label, chance)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    main_probes = (args.main_dir / "probes").resolve()
    dense_probes = (args.dense_dir / "probes").resolve()
    main_manifest = load_manifest(args.main_manifest if args.main_manifest else None)
    dense_manifest = load_manifest(args.dense_manifest if args.dense_manifest else None)
    print(f"[load] main probes from {main_probes} (manifest rows={0 if main_manifest is None else len(main_manifest)})")
    print(f"[load] dense probes from {dense_probes} (manifest rows={0 if dense_manifest is None else len(dense_manifest)})")

    main_df = collect_per_render(main_probes, "main", list(GLOBAL_TASKS.keys()), args.split)
    if not main_df.empty:
        main_df = merge_with_manifest(main_df, main_manifest)
    print(f"[shape] main per-render: {main_df.shape}")

    dense_df = collect_per_render(dense_probes, "dense", list(DENSE_TASKS.keys()), args.split)
    if not dense_df.empty:
        dense_df = merge_with_manifest(dense_df, dense_manifest)
    print(f"[shape] dense per-render: {dense_df.shape}")

    combined = pd.concat([main_df, dense_df], ignore_index=True, sort=False)
    save_dataframe(combined, out_dir / "state_space_per_render_table.csv")

    # Pick representative layers per task family for clarity:
    main_layers = {"final", "layer8"}
    dense_layers = {"final", "layer8"}
    for task in GLOBAL_TASKS:
        for layer in main_layers:
            state_space_for_task(main_df, task, layer, out_dir)
            print(f"[done] {task} ({layer}) state-space figures")
    for task in DENSE_TASKS:
        for layer in dense_layers:
            state_space_for_task(dense_df, task, layer, out_dir)
            print(f"[done] {task} ({layer}) state-space figures")

    print(f"[ok] figures written to: {out_dir}")


if __name__ == "__main__":
    main()

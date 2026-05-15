#!/usr/bin/env python3
"""Aggregate-style figures for the rerendered Experiment 1 results.

Produces:
  * CLIP-vs-DINO bar chart per task, faceted by texture condition.
  * Texture-sensitivity figure: photorealistic / flat / random_noise per model+task.
  * Layer-wise probe figure: metric vs layer for each model+task+texture.
  * Cross-texture transfer figure: train_X / test_Y heatmap per model+task.
  * Delta figure: DINO - CLIP and photoreal - {flat, random_noise}.
  * Aggregate robustness summary table (CSV + markdown).

Inputs (all configurable via CLI):
  --main-dir   Path to the main probe output tree (e.g. ``outputs/exp1_main``)
  --dense-dir  Path to the dense probe output tree (e.g. ``outputs/exp1_dense``)
  --out-dir    Combined output directory (default ``outputs/exp1_rerender_analysis_figures``)
  --split      Which split to plot (default ``test``)
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
    LAYER_ORDER,
    MODEL_COLORS,
    MODEL_DISPLAY,
    MODEL_FAMILY,
    TEXTURE_COLORS,
    TEXTURE_DISPLAY,
    TEXTURE_ORDER,
    TaskInfo,
    chance_line,
    filter_within_texture,
    get_pyplot,
    layer_sort_key,
    load_results,
    save_dataframe,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/exp1_rerender_analysis_figures"),
    )
    parser.add_argument("--split", default="test")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------


def primary_metric_table(
    main_results: pd.DataFrame,
    dense_results: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Pick the primary metric per task and return a tidy long-form table."""
    rows = []
    for results, family in ((main_results, "main"), (dense_results, "dense")):
        if results is None or results.empty:
            continue
        for task, info in ALL_TASKS.items():
            sub = results[
                (results["task"] == task)
                & (results["metric"] == info.primary_metric)
                & (results["split"] == split)
            ].copy()
            if sub.empty:
                continue
            sub["task_pretty"] = info.pretty
            sub["task_short"] = info.short
            sub["metric_direction"] = info.direction
            sub["chance"] = info.chance
            sub["probe_family"] = family
            rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["model_family"] = out["model"].map(MODEL_FAMILY)
    out["layer_order"] = out["layer"].map(lambda x: layer_sort_key(x)[0])
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _ordered_models(df: pd.DataFrame) -> List[str]:
    preferred = ["clip_vit_b16", "clip_vit_l14", "dinov2_vit_b", "dinov2_vit_l"]
    present = [m for m in preferred if m in df["model"].unique()]
    present.extend(sorted(m for m in df["model"].unique() if m not in present))
    return present


def _ordered_layers(df: pd.DataFrame) -> List[str]:
    layers = list(df["layer"].unique())
    return sorted(layers, key=layer_sort_key)


def _ordered_textures(df: pd.DataFrame) -> List[str]:
    return [t for t in TEXTURE_ORDER if t in df["texture_condition"].unique()]


# ---------------------------------------------------------------------------
# Figure 1: CLIP vs DINO aggregate (final-layer focus)
# ---------------------------------------------------------------------------


def plot_clip_vs_dino_final(table: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = table[(table["layer"] == "final") & (table["texture_condition"].isin(TEXTURE_ORDER))].copy()
    if sub.empty:
        return
    plt = get_pyplot()
    tasks = list(GLOBAL_TASKS.keys()) + list(DENSE_TASKS.keys())
    tasks = [t for t in tasks if t in sub["task"].unique()]
    models = _ordered_models(sub)
    textures = _ordered_textures(sub)

    n_tasks = len(tasks)
    fig, axes = plt.subplots(
        1, n_tasks, figsize=(3.2 * n_tasks + 1.2, 4.0), sharey=False
    )
    if n_tasks == 1:
        axes = [axes]
    rows_for_table: List[dict] = []
    for ax, task in zip(axes, tasks):
        info = ALL_TASKS[task]
        task_df = sub[sub["task"] == task]
        x_positions = np.arange(len(models))
        width = 0.27
        for i, tex in enumerate(textures):
            heights = []
            for model in models:
                row = task_df[(task_df["model"] == model) & (task_df["texture_condition"] == tex)]
                heights.append(row["value"].mean() if not row.empty else np.nan)
                rows_for_table.append(
                    dict(
                        task=task,
                        model=model,
                        texture_condition=tex,
                        value=row["value"].mean() if not row.empty else np.nan,
                        metric=info.primary_metric,
                        direction=info.direction,
                        chance=info.chance,
                    )
                )
            ax.bar(
                x_positions + (i - 1) * width,
                heights,
                width,
                color=TEXTURE_COLORS[tex],
                label=TEXTURE_DISPLAY[tex] if ax is axes[0] else None,
                edgecolor="black",
                linewidth=0.4,
            )
        if info.chance is not None:
            ax.axhline(
                info.chance,
                color="grey",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="chance" if ax is axes[0] else None,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=20, ha="right")
        ax.set_title(f"{info.pretty}\n[{info.primary_metric}] ({info.direction} = better)", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("primary metric")
    fig.suptitle(
        f"Aggregate CLIP vs DINO comparison (split={split}, final-layer features)",
        fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            bbox_to_anchor=(0.5, -0.05),
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, out_dir / "fig01_clip_vs_dino_final.png")
    save_dataframe(pd.DataFrame(rows_for_table), out_dir / "fig01_clip_vs_dino_final.csv")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Texture sensitivity (photoreal -> flat / random) per model/task
# ---------------------------------------------------------------------------


def plot_texture_sensitivity(table: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = table[(table["layer"] == "final") & (table["texture_condition"].isin(TEXTURE_ORDER))].copy()
    if sub.empty:
        return
    plt = get_pyplot()
    tasks = [t for t in list(GLOBAL_TASKS.keys()) + list(DENSE_TASKS.keys()) if t in sub["task"].unique()]
    models = _ordered_models(sub)
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(3.2 * len(tasks) + 1.5, 4.0), sharey=False
    )
    if len(tasks) == 1:
        axes = [axes]
    rows_for_table: List[dict] = []
    for ax, task in zip(axes, tasks):
        info = ALL_TASKS[task]
        for model in models:
            ys: List[float] = []
            xs: List[str] = []
            for tex in TEXTURE_ORDER:
                row = sub[(sub["task"] == task) & (sub["model"] == model) & (sub["texture_condition"] == tex)]
                if row.empty:
                    continue
                val = row["value"].mean()
                ys.append(val)
                xs.append(tex)
                rows_for_table.append(
                    dict(task=task, model=model, texture_condition=tex, value=val, metric=info.primary_metric)
                )
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                color=MODEL_COLORS.get(model, None),
                label=MODEL_DISPLAY.get(model, model) if ax is axes[0] else None,
                linewidth=2,
            )
        if info.chance is not None:
            ax.axhline(info.chance, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(range(len(TEXTURE_ORDER)))
        ax.set_xticklabels([TEXTURE_DISPLAY[t] for t in TEXTURE_ORDER], rotation=15)
        ax.set_title(f"{info.pretty}\n[{info.primary_metric}] ({info.direction} = better)", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("primary metric")
    fig.suptitle(f"Texture sensitivity (final-layer, split={split})", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            bbox_to_anchor=(0.5, -0.05),
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, out_dir / "fig02_texture_sensitivity.png")
    save_dataframe(pd.DataFrame(rows_for_table), out_dir / "fig02_texture_sensitivity.csv")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Layer-wise probe figure
# ---------------------------------------------------------------------------


def plot_layerwise(table: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = table[table["texture_condition"].isin(TEXTURE_ORDER)].copy()
    if sub.empty:
        return
    plt = get_pyplot()
    tasks = [t for t in list(GLOBAL_TASKS.keys()) + list(DENSE_TASKS.keys()) if t in sub["task"].unique()]
    textures = _ordered_textures(sub)
    models = _ordered_models(sub)

    n_rows = len(tasks)
    n_cols = len(textures)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows), sharex=False, sharey=False
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    rows_for_table: List[dict] = []
    for r, task in enumerate(tasks):
        info = ALL_TASKS[task]
        for c, tex in enumerate(textures):
            ax = axes[r, c]
            for model in models:
                rows = sub[(sub["task"] == task) & (sub["model"] == model) & (sub["texture_condition"] == tex)]
                if rows.empty:
                    continue
                rows = rows.sort_values("layer_order")
                ax.plot(
                    rows["layer"],
                    rows["value"],
                    marker="o",
                    color=MODEL_COLORS.get(model, None),
                    label=MODEL_DISPLAY.get(model, model) if r == 0 and c == 0 else None,
                    linewidth=2,
                )
                for _, row in rows.iterrows():
                    rows_for_table.append(
                        dict(
                            task=task,
                            model=model,
                            texture_condition=tex,
                            layer=row["layer"],
                            value=row["value"],
                            metric=info.primary_metric,
                            direction=info.direction,
                        )
                    )
            if info.chance is not None:
                ax.axhline(info.chance, color="grey", linestyle="--", linewidth=1, alpha=0.7)
            if r == 0:
                ax.set_title(TEXTURE_DISPLAY[tex])
            if c == 0:
                ax.set_ylabel(f"{info.short}\n[{info.primary_metric}]", fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=8,
        )
    fig.suptitle(
        f"Layer-wise probe performance (split={split}, lower is better for *_error/MAE metrics)",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, out_dir / "fig03_layerwise.png")
    save_dataframe(pd.DataFrame(rows_for_table), out_dir / "fig03_layerwise.csv")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Cross-texture transfer
# ---------------------------------------------------------------------------


def plot_cross_texture_transfer(
    main_results: pd.DataFrame, out_dir: Path, split: str
) -> None:
    if main_results.empty:
        return
    df = main_results[main_results["split"] == split].copy()
    if df.empty:
        return
    plt = get_pyplot()
    tasks = [t for t in GLOBAL_TASKS if t in df["task"].unique()]
    models = _ordered_models(df)
    rows_for_table: List[dict] = []
    for task in tasks:
        info = GLOBAL_TASKS[task]
        sub = df[(df["task"] == task) & (df["metric"] == info.primary_metric) & (df["layer"] == "final")]
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(2.6 * n_models + 1.0, 3.0))
        if n_models == 1:
            axes = [axes]
        vmin, vmax = None, None
        # Pre-compute global value range for consistent colormap
        vals = []
        for m in models:
            for train_tex in TEXTURE_ORDER:
                for test_tex in TEXTURE_ORDER:
                    cond = (
                        f"texture_{train_tex}"
                        if train_tex == test_tex
                        else f"train_{train_tex}__test_{test_tex}"
                    )
                    raw_cond = train_tex if train_tex == test_tex else f"train_{train_tex}__test_{test_tex}"
                    row = sub[(sub["model"] == m) & (sub["texture_condition"] == raw_cond)]
                    if not row.empty:
                        vals.append(row["value"].mean())
        if vals:
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
        for ax, model in zip(axes, models):
            matrix = np.full((len(TEXTURE_ORDER), len(TEXTURE_ORDER)), np.nan)
            for i, train_tex in enumerate(TEXTURE_ORDER):
                for j, test_tex in enumerate(TEXTURE_ORDER):
                    raw_cond = train_tex if train_tex == test_tex else f"train_{train_tex}__test_{test_tex}"
                    row = sub[(sub["model"] == model) & (sub["texture_condition"] == raw_cond)]
                    if row.empty:
                        continue
                    val = row["value"].mean()
                    matrix[i, j] = val
                    rows_for_table.append(
                        dict(
                            task=task,
                            model=model,
                            train_texture=train_tex,
                            test_texture=test_tex,
                            value=val,
                            metric=info.primary_metric,
                            direction=info.direction,
                        )
                    )
            cmap = "viridis" if info.direction == "higher" else "viridis_r"
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if not np.isnan(matrix[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{matrix[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="white" if info.direction == "lower" else "black",
                            fontsize=9,
                        )
            ax.set_xticks(range(len(TEXTURE_ORDER)))
            ax.set_yticks(range(len(TEXTURE_ORDER)))
            ax.set_xticklabels([TEXTURE_DISPLAY[t] for t in TEXTURE_ORDER], rotation=20)
            ax.set_yticklabels([TEXTURE_DISPLAY[t] for t in TEXTURE_ORDER])
            ax.set_xlabel("test texture")
            ax.set_ylabel("train texture")
            ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"Train/test texture transfer ({info.pretty}, split={split}, final-layer)\n[{info.primary_metric}] ({info.direction} = better)",
            fontsize=11,
        )
        fig.tight_layout()
        save_figure(fig, out_dir / f"fig04_cross_texture_transfer_{info.short}.png")
        plt.close(fig)
    save_dataframe(pd.DataFrame(rows_for_table), out_dir / "fig04_cross_texture_transfer.csv")


# ---------------------------------------------------------------------------
# Figure 5: Delta DINO - CLIP and photoreal - {flat, random_noise}
# ---------------------------------------------------------------------------


def _signed_advantage(values: pd.Series, direction: str) -> pd.Series:
    """For a 'lower is better' metric, lower values are good, so subtract reversed."""
    return values if direction == "higher" else -values


def plot_deltas(table: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = table[(table["layer"] == "final") & (table["texture_condition"].isin(TEXTURE_ORDER))].copy()
    if sub.empty:
        return
    plt = get_pyplot()
    tasks = [t for t in list(GLOBAL_TASKS.keys()) + list(DENSE_TASKS.keys()) if t in sub["task"].unique()]
    textures = _ordered_textures(sub)
    delta_rows: List[dict] = []
    for task in tasks:
        info = ALL_TASKS[task]
        rep_dino = "dinov2_vit_b" if "dinov2_vit_b" in sub.model.unique() else None
        rep_clip = "clip_vit_b16" if "clip_vit_b16" in sub.model.unique() else None
        if rep_dino is None or rep_clip is None:
            continue
        for tex in textures:
            d = sub[(sub["task"] == task) & (sub["texture_condition"] == tex) & (sub["model"] == rep_dino)]["value"]
            c = sub[(sub["task"] == task) & (sub["texture_condition"] == tex) & (sub["model"] == rep_clip)]["value"]
            if d.empty or c.empty:
                continue
            d_val, c_val = float(d.mean()), float(c.mean())
            advantage = (d_val - c_val) if info.direction == "higher" else (c_val - d_val)
            delta_rows.append(
                dict(
                    task=task,
                    texture_condition=tex,
                    dino_value=d_val,
                    clip_value=c_val,
                    dino_minus_clip_better=advantage,
                    metric=info.primary_metric,
                    direction=info.direction,
                )
            )
    delta_df = pd.DataFrame(delta_rows)
    if delta_df.empty:
        return
    fig, ax = plt.subplots(figsize=(1.8 * len(tasks) + 1.0, 4.0))
    pivot = delta_df.pivot(index="task", columns="texture_condition", values="dino_minus_clip_better")
    pivot = pivot.reindex(index=tasks, columns=[t for t in TEXTURE_ORDER if t in pivot.columns])
    x = np.arange(len(pivot.index))
    width = 0.27
    for i, tex in enumerate(pivot.columns):
        ax.bar(x + (i - 1) * width, pivot[tex].values, width, color=TEXTURE_COLORS[tex], label=TEXTURE_DISPLAY[tex], edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([ALL_TASKS[t].short for t in pivot.index], rotation=20)
    ax.set_ylabel("DINO advantage over CLIP\n(higher = DINO better, in direction-aware metric units)")
    ax.set_title(f"DINOv2 ViT-B minus CLIP ViT-B/16 (final-layer, split={split})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig05a_delta_dino_minus_clip.png")
    save_dataframe(delta_df, out_dir / "fig05a_delta_dino_minus_clip.csv")
    plt.close(fig)

    # photoreal -> flat / random delta per model
    rows: List[dict] = []
    for task in tasks:
        info = ALL_TASKS[task]
        for model in _ordered_models(sub):
            ph = sub[(sub["task"] == task) & (sub["model"] == model) & (sub["texture_condition"] == "photorealistic")]
            if ph.empty:
                continue
            ph_v = float(ph["value"].mean())
            for cmp_tex in ("flat", "random_noise"):
                rr = sub[(sub["task"] == task) & (sub["model"] == model) & (sub["texture_condition"] == cmp_tex)]
                if rr.empty:
                    continue
                cmp_v = float(rr["value"].mean())
                # texture drop = how much performance declined from photorealistic
                drop = (ph_v - cmp_v) if info.direction == "higher" else (cmp_v - ph_v)
                rows.append(
                    dict(
                        task=task,
                        model=model,
                        comparison_texture=cmp_tex,
                        photoreal_value=ph_v,
                        comparison_value=cmp_v,
                        drop=drop,
                        metric=info.primary_metric,
                        direction=info.direction,
                    )
                )
    drop_df = pd.DataFrame(rows)
    if drop_df.empty:
        return
    # Per-task subplot (own y-axis) so very-different metric units do not
    # collapse into a single mis-leading bar height.
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.0 * n_tasks + 1.0, 4.0))
    if n_tasks == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        info = ALL_TASKS[task]
        sub2 = drop_df[drop_df["task"] == task]
        models_in = [m for m in _ordered_models(sub) if m in sub2["model"].unique()]
        cmps = ["flat", "random_noise"]
        width = 0.8 / max(len(models_in), 1)
        x = np.arange(len(cmps))
        for i, model in enumerate(models_in):
            heights = []
            for cmp_tex in cmps:
                row = sub2[(sub2["model"] == model) & (sub2["comparison_texture"] == cmp_tex)]
                heights.append(row["drop"].iloc[0] if not row.empty else np.nan)
            ax.bar(
                x + (i - (len(models_in) - 1) / 2) * width,
                heights,
                width,
                color=MODEL_COLORS.get(model, None),
                label=MODEL_DISPLAY.get(model, model) if ax is axes[0] else None,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"→ {TEXTURE_DISPLAY[t]}" for t in cmps], rotation=10)
        ax.set_title(f"{info.short}\n[{info.primary_metric}] ({info.direction} = better)", fontsize=9)
        ax.set_ylabel("drop vs photoreal\n(positive = worse)", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.05), fontsize=8
        )
    fig.suptitle(
        f"Texture removal: per-task degradation vs photoreal (split={split})", fontsize=12
    )
    fig.tight_layout()
    save_figure(fig, out_dir / "fig05b_texture_drop_per_model.png")
    save_dataframe(drop_df, out_dir / "fig05b_texture_drop_per_model.csv")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Per-render summary (fraction positive / above chance)
# ---------------------------------------------------------------------------


def _per_render_signal(probe_root: Path, family: str, tasks, split: str) -> pd.DataFrame:
    """Return summary stats per (model, layer, task, texture) for per-render scores."""
    from scripts.analysis._common import load_per_render_table

    df = load_per_render_table(probe_root, family=family, tasks=tasks)
    if df.empty:
        return df
    df = df[df["split"] == split].copy()
    rows: List[dict] = []
    for (model, layer, task, tex), group in df.groupby(["model", "layer", "task", "texture_condition"]):
        info = ALL_TASKS.get(task)
        if info is None or info.per_render_column is None:
            continue
        if info.per_render_column not in group.columns:
            continue
        vals = group[info.per_render_column].astype(float).values
        if len(vals) == 0 or np.all(np.isnan(vals)):
            continue
        vals = vals[~np.isnan(vals)]
        higher = info.per_render_direction == "higher"
        chance = info.chance if higher else None
        row = dict(
            model=model,
            layer=layer,
            task=task,
            texture_condition=tex,
            n=len(vals),
            mean=float(np.mean(vals)),
            median=float(np.median(vals)),
            std=float(np.std(vals)),
            p10=float(np.percentile(vals, 10)),
            p25=float(np.percentile(vals, 25)),
            p75=float(np.percentile(vals, 75)),
            p90=float(np.percentile(vals, 90)),
        )
        if higher and chance is not None:
            row["frac_above_chance"] = float(np.mean(vals > chance))
        if higher and info.chance == 0.0:
            row["frac_positive"] = float(np.mean(vals > 0.0))
        if not higher and info.chance is not None:
            row["frac_below_chance"] = float(np.mean(vals < info.chance))
        row["metric"] = info.per_render_column
        row["direction"] = info.per_render_direction
        row["chance"] = info.chance
        rows.append(row)
    return pd.DataFrame(rows)


def plot_per_render_summary(
    main_probe_root: Path, dense_probe_root: Path, out_dir: Path, split: str
) -> None:
    main_df = _per_render_signal(main_probe_root, "main", list(GLOBAL_TASKS.keys()), split)
    dense_df = _per_render_signal(dense_probe_root, "dense", list(DENSE_TASKS.keys()), split)
    summary = pd.concat([main_df, dense_df], ignore_index=True, sort=False)
    if summary.empty:
        return
    save_dataframe(summary, out_dir / "fig06_per_render_summary_table.csv")

    plt = get_pyplot()
    # subset to final-layer for the bar plot
    sub = summary[summary["layer"] == "final"].copy()
    if sub.empty:
        return
    tasks = sorted(sub["task"].unique(), key=lambda t: list(ALL_TASKS.keys()).index(t) if t in ALL_TASKS else 99)
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.0 * n_tasks + 1.2, 3.8))
    if n_tasks == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        info = ALL_TASKS[task]
        block = sub[sub["task"] == task]
        models = _ordered_models(block)
        textures = _ordered_textures(block)
        x = np.arange(len(models))
        width = 0.27
        for i, tex in enumerate(textures):
            heights = []
            for m in models:
                row = block[(block["model"] == m) & (block["texture_condition"] == tex)]
                if row.empty:
                    heights.append(np.nan)
                    continue
                if info.per_render_direction == "higher" and info.chance is not None:
                    heights.append(row["frac_above_chance"].iloc[0])
                elif info.per_render_direction == "lower" and info.chance is not None:
                    heights.append(row["frac_below_chance"].iloc[0])
                else:
                    heights.append(np.nan)
            ax.bar(
                x + (i - 1) * width,
                heights,
                width,
                color=TEXTURE_COLORS[tex],
                label=TEXTURE_DISPLAY[tex] if ax is axes[0] else None,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=20, ha="right")
        cmp = "above" if info.per_render_direction == "higher" else "below"
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"{info.pretty}\nfraction of renders {cmp} chance ({info.chance})", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("fraction of test renders")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.05), fontsize=8
        )
    fig.suptitle(
        f"Per-render reliability: how often does the probe beat chance? (split={split}, final-layer)",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, out_dir / "fig06_per_render_above_chance.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Main (global pairwise depth) vs Dense (per-patch depth) comparison
# ---------------------------------------------------------------------------


def plot_main_vs_dense(
    main_results: pd.DataFrame, dense_results: pd.DataFrame, out_dir: Path, split: str
) -> None:
    """Show that global and dense depth probes measure different accessible signals.

    Left panel: global pairwise relative-depth accuracy (main run, ``relative_depth_regions``).
    Right panel: dense per-render Pearson r mean (dense run, ``dense_depth_patches``).
    Both panels are at the final layer and ``texture_condition in TEXTURE_ORDER``.
    """
    if main_results.empty or dense_results.empty:
        return
    plt = get_pyplot()
    main_sub = main_results[
        (main_results["split"] == split)
        & (main_results["task"] == "relative_depth_regions")
        & (main_results["metric"] == "valid_pair_accuracy")
        & (main_results["layer"] == "final")
        & (main_results["texture_condition"].isin(TEXTURE_ORDER))
    ].copy()
    dense_sub = dense_results[
        (dense_results["split"] == split)
        & (dense_results["task"] == "dense_depth_patches")
        & (dense_results["metric"] == "pearson_r_mean")
        & (dense_results["layer"] == "final")
        & (dense_results["texture_condition"].isin(TEXTURE_ORDER))
    ].copy()
    if main_sub.empty or dense_sub.empty:
        return
    models_main = _ordered_models(main_sub)
    models_dense = _ordered_models(dense_sub)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
    rows_for_table: List[dict] = []

    def _bar(ax, sub, models, title, ylabel, chance):
        x = np.arange(len(models))
        width = 0.27
        for i, tex in enumerate([t for t in TEXTURE_ORDER if t in sub["texture_condition"].unique()]):
            heights = []
            for m in models:
                row = sub[(sub["model"] == m) & (sub["texture_condition"] == tex)]
                heights.append(row["value"].mean() if not row.empty else np.nan)
                rows_for_table.append(
                    dict(panel=title, model=m, texture_condition=tex, value=row["value"].mean() if not row.empty else np.nan)
                )
            ax.bar(
                x + (i - 1) * width,
                heights,
                width,
                color=TEXTURE_COLORS[tex],
                label=TEXTURE_DISPLAY[tex] if title.startswith("Global") else None,
                edgecolor="black",
                linewidth=0.4,
            )
        if chance is not None:
            ax.axhline(chance, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    _bar(
        axes[0],
        main_sub,
        models_main,
        "Global probe — relative depth (region-pair accuracy)\n(main rerender, valid_pair_accuracy, higher = better)",
        "valid pair accuracy",
        0.5,
    )
    _bar(
        axes[1],
        dense_sub,
        models_dense,
        "Dense probe — per-render Pearson r mean\n(dense rerender, pearson_r_mean, higher = better)",
        "Pearson r (mean over test renders)",
        0.0,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            bbox_to_anchor=(0.5, -0.05),
            fontsize=8,
        )
    fig.suptitle(
        f"Global vs dense depth probes (split={split}). The two panels measure different accessible signals from\n"
        "the same frozen features and are not on the same scale; they are presented side-by-side to highlight that\n"
        "DINO's dense advantage is much larger than its global-relative-depth advantage.",
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, out_dir / "fig07_main_vs_dense_depth.png")
    save_dataframe(pd.DataFrame(rows_for_table), out_dir / "fig07_main_vs_dense_depth.csv")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate robustness table
# ---------------------------------------------------------------------------


def write_robustness_table(
    main_results: pd.DataFrame,
    dense_results: pd.DataFrame,
    main_probe_root: Path,
    dense_probe_root: Path,
    out_dir: Path,
    split: str,
) -> None:
    """Write a compact summary CSV/markdown with primary metric + per-render stats."""
    primary = primary_metric_table(main_results, dense_results, split)
    primary = primary[primary["texture_condition"].isin(TEXTURE_ORDER)].copy()
    if primary.empty:
        return
    # per-render summary
    pr_main = _per_render_signal(main_probe_root, "main", list(GLOBAL_TASKS.keys()), split)
    pr_dense = _per_render_signal(dense_probe_root, "dense", list(DENSE_TASKS.keys()), split)
    pr = pd.concat([pr_main, pr_dense], ignore_index=True, sort=False)
    pr_compact = pr.rename(
        columns={
            "mean": "per_render_mean",
            "median": "per_render_median",
            "std": "per_render_std",
            "n": "n_renders",
        }
    )
    keep_cols = [
        "model",
        "layer",
        "task",
        "texture_condition",
        "n_renders",
        "per_render_mean",
        "per_render_median",
        "per_render_std",
        "frac_above_chance",
        "frac_below_chance",
        "frac_positive",
    ]
    keep_cols = [c for c in keep_cols if c in pr_compact.columns]
    pr_compact = pr_compact[keep_cols]

    merged = primary[
        ["task", "task_pretty", "model", "layer", "texture_condition", "metric", "value", "metric_direction", "chance"]
    ].merge(
        pr_compact,
        on=["model", "layer", "task", "texture_condition"],
        how="left",
    )
    merged = merged.sort_values(["task", "model", "layer", "texture_condition"]).reset_index(drop=True)
    save_dataframe(merged, out_dir / "robustness_summary.csv")

    md_lines = [
        "# Robustness summary (rerendered Experiment 1)",
        "",
        f"Split: ``{split}``. ``metric`` is the per-task primary aggregate metric. "
        f"``per_render_*`` columns summarize the per-render score (from each probe's predictions.csv).",
        "",
        "| task | model | layer | texture | metric | value | per_render_mean | per_render_median | n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    final_only = merged[merged["layer"] == "final"]
    for _, r in final_only.iterrows():
        md_lines.append(
            "| {task} | {model} | {layer} | {tex} | {metric} ({dir}) | {value:.4f} | {mean} | {median} | {n} |".format(
                task=r["task"],
                model=r["model"],
                layer=r["layer"],
                tex=r["texture_condition"],
                metric=r["metric"],
                dir=r["metric_direction"],
                value=float(r["value"]) if pd.notna(r["value"]) else float("nan"),
                mean=f"{r['per_render_mean']:.4f}" if pd.notna(r.get("per_render_mean")) else "-",
                median=f"{r['per_render_median']:.4f}" if pd.notna(r.get("per_render_median")) else "-",
                n=int(r["n_renders"]) if pd.notna(r.get("n_renders")) else 0,
            )
        )
    (out_dir / "robustness_summary.md").write_text("\n".join(md_lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    main_results_path = args.main_dir / "results/exp1_results_long.csv"
    dense_results_path = args.dense_dir / "results/exp1_results_long.csv"
    print(f"[load] {main_results_path}")
    main_results = load_results(main_results_path)
    print(f"[load] {dense_results_path}")
    dense_results = load_results(dense_results_path)

    main_probe_root = (args.main_dir / "probes").resolve()
    dense_probe_root = (args.dense_dir / "probes").resolve()

    table = primary_metric_table(main_results, dense_results, args.split)
    print(f"[shape] primary metric table: {table.shape}")

    plot_clip_vs_dino_final(table, out_dir, args.split)
    print("[done] fig01_clip_vs_dino_final")
    plot_texture_sensitivity(table, out_dir, args.split)
    print("[done] fig02_texture_sensitivity")
    plot_layerwise(table, out_dir, args.split)
    print("[done] fig03_layerwise")
    plot_cross_texture_transfer(main_results, out_dir, args.split)
    print("[done] fig04_cross_texture_transfer_*")
    plot_deltas(table, out_dir, args.split)
    print("[done] fig05a_delta_dino_minus_clip / fig05b_texture_drop_per_model")
    plot_per_render_summary(main_probe_root, dense_probe_root, out_dir, args.split)
    print("[done] fig06_per_render_above_chance")
    plot_main_vs_dense(main_results, dense_results, out_dir, args.split)
    print("[done] fig07_main_vs_dense")
    write_robustness_table(main_results, dense_results, main_probe_root, dense_probe_root, out_dir, args.split)
    print("[done] robustness_summary.csv/.md")

    print(f"[ok] figures written to: {out_dir}")


if __name__ == "__main__":
    main()

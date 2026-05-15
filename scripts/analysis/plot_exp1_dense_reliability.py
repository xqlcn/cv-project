#!/usr/bin/env python3
"""Per-render reliability sweep for the dense rerendered Experiment 1 probes.

Loads the per-render Pearson r / RMSE / angular-error rows from each
``predictions.csv`` and the per-patch prediction tensors from
``predictions_<split>.npz`` to compute additional reliability summaries that
the aggregate single-number metric alone cannot reveal.

Generates:
  * Distribution figures (violin + ECDF + histogram) of per-render Pearson r
    for dense depth, by model / layer / texture.
  * Distribution figures of per-render angular error (median) for dense
    surface normals.
  * Mean / median / fraction-positive / fraction-above-threshold summary
    tables, broken out by model / layer / texture / split.
  * A scatter contrasting CLIP vs DINO per-render scores on matched renders.

Higher Pearson r is better; chance is 0. Lower angular error is better;
chance for uniformly random unit-vector predictions is ~90 degrees.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis._common import (  # noqa: E402
    DENSE_TASKS,
    MODEL_COLORS,
    MODEL_DISPLAY,
    TEXTURE_COLORS,
    TEXTURE_DISPLAY,
    TEXTURE_ORDER,
    get_pyplot,
    layer_sort_key,
    load_dense_npz,
    load_per_render_table,
    save_dataframe,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/exp1_rerender_analysis_figures/dense_reliability"),
    )
    parser.add_argument("--split", default="test")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loader: per-render table for dense probes
# ---------------------------------------------------------------------------


def collect_dense_per_render(probe_root: Path, split: str) -> pd.DataFrame:
    df = load_per_render_table(
        probe_root,
        family="dense",
        tasks=list(DENSE_TASKS.keys()),
    )
    if df.empty:
        return df
    return df[df["split"] == split].copy()


def per_render_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if df.empty:
        return pd.DataFrame()
    for (model, layer, task, tex), grp in df.groupby(
        ["model", "layer", "task", "texture_condition"]
    ):
        info = DENSE_TASKS[task]
        col = info.per_render_column
        if col not in grp.columns:
            continue
        vals = grp[col].astype(float).dropna().values
        if len(vals) == 0:
            continue
        row: Dict[str, float] = dict(
            model=model,
            layer=layer,
            task=task,
            texture_condition=tex,
            n=int(len(vals)),
            mean=float(np.mean(vals)),
            median=float(np.median(vals)),
            std=float(np.std(vals)),
            p10=float(np.percentile(vals, 10)),
            p25=float(np.percentile(vals, 25)),
            p75=float(np.percentile(vals, 75)),
            p90=float(np.percentile(vals, 90)),
        )
        if task == "dense_depth_patches":
            row["frac_positive"] = float(np.mean(vals > 0.0))
            row["frac_above_0_2"] = float(np.mean(vals > 0.2))
            row["frac_above_0_4"] = float(np.mean(vals > 0.4))
        elif task == "dense_surface_normal_patches":
            row["frac_below_60_deg"] = float(np.mean(vals < 60.0))
            row["frac_below_45_deg"] = float(np.mean(vals < 45.0))
            row["frac_below_30_deg"] = float(np.mean(vals < 30.0))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Recompute additional dense-depth metrics from NPZ (object-only correlation)
# ---------------------------------------------------------------------------


def recompute_per_render_dense_depth(
    probe_root: Path, split: str, models: Sequence[str], layers: Sequence[str]
) -> pd.DataFrame:
    """For each render, recompute Pearson r over valid patches and a few
    object-only-mask aware variants from the NPZ files."""
    rows: List[dict] = []
    for model in models:
        for layer in layers:
            for tex in TEXTURE_ORDER:
                data = load_dense_npz(
                    probe_root, model, layer, "dense_depth_patches", tex, split=split
                )
                if data is None:
                    continue
                preds = data["predictions"]
                targets = data["targets"]
                valid = data["valid"]
                render_ids = data["render_ids"]
                for i, rid in enumerate(render_ids):
                    p = preds[i].ravel()
                    t = targets[i].ravel()
                    v = valid[i].ravel().astype(bool)
                    if v.sum() < 3:
                        continue
                    p = p[v]
                    t = t[v]
                    if np.std(p) < 1e-8 or np.std(t) < 1e-8:
                        pearson = 0.0
                    else:
                        pearson = float(np.corrcoef(p, t)[0, 1])
                    abs_err = np.abs(p - t)
                    rows.append(
                        dict(
                            model=model,
                            layer=layer,
                            task="dense_depth_patches",
                            texture_condition=tex,
                            split=split,
                            render_id=str(rid),
                            n_valid_patches=int(v.sum()),
                            pearson_r_object=pearson,
                            mae_object=float(np.mean(abs_err)),
                            rmse_object=float(np.sqrt(np.mean(abs_err ** 2))),
                        )
                    )
    return pd.DataFrame(rows)


def recompute_per_render_dense_normals(
    probe_root: Path, split: str, models: Sequence[str], layers: Sequence[str]
) -> pd.DataFrame:
    """Recompute per-render angular error stats restricted to valid (foreground)
    patches for dense normals."""
    rows: List[dict] = []
    for model in models:
        for layer in layers:
            for tex in TEXTURE_ORDER:
                data = load_dense_npz(
                    probe_root, model, layer, "dense_surface_normal_patches", tex, split=split
                )
                if data is None:
                    continue
                preds = data["predictions"]  # (N, H, W, 3)
                targets = data["targets"]
                valid = data["valid"]  # (N, H, W)
                render_ids = data["render_ids"]
                for i, rid in enumerate(render_ids):
                    v = valid[i].astype(bool)
                    if v.sum() < 3:
                        continue
                    p = preds[i][v]
                    t = targets[i][v]
                    p_norm = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
                    t_norm = t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)
                    cos = np.clip(np.sum(p_norm * t_norm, axis=-1), -1.0, 1.0)
                    ang = np.degrees(np.arccos(cos))
                    rows.append(
                        dict(
                            model=model,
                            layer=layer,
                            task="dense_surface_normal_patches",
                            texture_condition=tex,
                            split=split,
                            render_id=str(rid),
                            n_valid_patches=int(v.sum()),
                            angular_err_mean_object=float(np.mean(ang)),
                            angular_err_median_object=float(np.median(ang)),
                            within_30_object=float(np.mean(ang < 30.0)),
                            within_45_object=float(np.mean(ang < 45.0)),
                            within_60_object=float(np.mean(ang < 60.0)),
                        )
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _ordered_models(df: pd.DataFrame) -> List[str]:
    preferred = ["clip_vit_b16", "clip_vit_l14", "dinov2_vit_b", "dinov2_vit_l"]
    pres = [m for m in preferred if m in df["model"].unique()]
    pres.extend(sorted(m for m in df["model"].unique() if m not in pres))
    return pres


def _ordered_layers(df: pd.DataFrame) -> List[str]:
    return sorted(df["layer"].unique(), key=layer_sort_key)


def plot_pearson_distributions(per_render: pd.DataFrame, out_dir: Path) -> None:
    df = per_render[per_render["task"] == "dense_depth_patches"].copy()
    if df.empty:
        return
    plt = get_pyplot()
    layers = _ordered_layers(df)
    models = _ordered_models(df)
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.0 * n_layers + 1.0, 4.0), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        sub = df[df["layer"] == layer]
        positions = []
        labels: List[str] = []
        pos_i = 0
        for m in models:
            for tex in TEXTURE_ORDER:
                vals = sub[(sub["model"] == m) & (sub["texture_condition"] == tex)]["pearson_r"].dropna().values
                if len(vals) == 0:
                    continue
                pos_i += 1
                positions.append(pos_i)
                labels.append(f"{MODEL_DISPLAY.get(m, m)}\n{TEXTURE_DISPLAY[tex]}")
                parts = ax.violinplot(
                    [vals],
                    positions=[pos_i],
                    widths=0.7,
                    showmeans=False,
                    showmedians=True,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(TEXTURE_COLORS[tex])
                    body.set_edgecolor(MODEL_COLORS.get(m, "black"))
                    body.set_alpha(0.6)
                # individual points (jittered)
                jitter = np.random.RandomState(0).uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    np.full_like(vals, pos_i) + jitter,
                    vals,
                    s=8,
                    alpha=0.4,
                    color="black",
                )
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"layer = {layer}")
        ax.set_ylabel("per-render Pearson r (predictions.csv)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Dense depth: distribution of per-render Pearson r (higher = better, chance = 0)")
    fig.tight_layout()
    save_figure(fig, out_dir / "dense_depth_pearson_distribution.png")
    plt.close(fig)


def plot_pearson_ecdf(per_render: pd.DataFrame, out_dir: Path) -> None:
    df = per_render[per_render["task"] == "dense_depth_patches"].copy()
    if df.empty:
        return
    plt = get_pyplot()
    layers = _ordered_layers(df)
    fig, axes = plt.subplots(1, len(layers), figsize=(4.0 * len(layers) + 1.0, 4.0), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        sub = df[df["layer"] == layer]
        for m in _ordered_models(sub):
            for tex in TEXTURE_ORDER:
                vals = sub[(sub["model"] == m) & (sub["texture_condition"] == tex)]["pearson_r"].dropna().values
                if len(vals) == 0:
                    continue
                xs = np.sort(vals)
                ys = np.arange(1, len(xs) + 1) / len(xs)
                ax.step(
                    xs,
                    ys,
                    where="post",
                    color=MODEL_COLORS.get(m, "k"),
                    linestyle={"photorealistic": "-", "flat": "--", "random_noise": ":"}[tex],
                    label=f"{MODEL_DISPLAY.get(m, m)} / {TEXTURE_DISPLAY[tex]}",
                )
        ax.axvline(0.0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("per-render Pearson r")
        ax.set_ylabel("ECDF")
        ax.set_title(f"layer = {layer}")
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Dense depth: ECDF of per-render Pearson r per model / layer / texture")
    fig.tight_layout()
    save_figure(fig, out_dir / "dense_depth_pearson_ecdf.png")
    plt.close(fig)


def plot_angular_error_distributions(per_render: pd.DataFrame, out_dir: Path) -> None:
    df = per_render[per_render["task"] == "dense_surface_normal_patches"].copy()
    if df.empty:
        return
    plt = get_pyplot()
    layers = _ordered_layers(df)
    models = _ordered_models(df)
    fig, axes = plt.subplots(1, len(layers), figsize=(4.0 * len(layers) + 1.0, 4.0), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        sub = df[df["layer"] == layer]
        positions: List[int] = []
        labels: List[str] = []
        pos_i = 0
        for m in models:
            for tex in TEXTURE_ORDER:
                vals = sub[(sub["model"] == m) & (sub["texture_condition"] == tex)]["angular_error_deg_median"].dropna().values
                if len(vals) == 0:
                    continue
                pos_i += 1
                positions.append(pos_i)
                labels.append(f"{MODEL_DISPLAY.get(m, m)}\n{TEXTURE_DISPLAY[tex]}")
                parts = ax.violinplot(
                    [vals],
                    positions=[pos_i],
                    widths=0.7,
                    showmeans=False,
                    showmedians=True,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(TEXTURE_COLORS[tex])
                    body.set_edgecolor(MODEL_COLORS.get(m, "black"))
                    body.set_alpha(0.6)
                jitter = np.random.RandomState(0).uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    np.full_like(vals, pos_i) + jitter,
                    vals,
                    s=8,
                    alpha=0.4,
                    color="black",
                )
        ax.axhline(90.0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"layer = {layer}")
        ax.set_ylabel("per-render median angular error (deg)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Dense surface normals: per-render median angular error (lower = better, chance ≈ 90°)")
    fig.tight_layout()
    save_figure(fig, out_dir / "dense_normals_angerr_distribution.png")
    plt.close(fig)


def plot_object_only_pearson(object_only: pd.DataFrame, out_dir: Path) -> None:
    """Show CLIP vs DINO per-render object-only Pearson r as a scatter."""
    if object_only.empty:
        return
    df = object_only[object_only["task"] == "dense_depth_patches"].copy()
    if df.empty:
        return
    plt = get_pyplot()
    layers = _ordered_layers(df)
    pairs = [("clip_vit_b16", "dinov2_vit_b")]
    rows_for_table: List[dict] = []
    fig, axes = plt.subplots(
        len(pairs), len(layers), figsize=(4.0 * len(layers) + 1.0, 4.0 * len(pairs)), squeeze=False
    )
    for r, (clip_m, dino_m) in enumerate(pairs):
        for c, layer in enumerate(layers):
            ax = axes[r, c]
            sub = df[df["layer"] == layer]
            clip_df = sub[sub["model"] == clip_m][["render_id", "texture_condition", "pearson_r_object"]].rename(
                columns={"pearson_r_object": "clip"}
            )
            dino_df = sub[sub["model"] == dino_m][["render_id", "texture_condition", "pearson_r_object"]].rename(
                columns={"pearson_r_object": "dino"}
            )
            merged = clip_df.merge(dino_df, on=["render_id", "texture_condition"], how="inner")
            if merged.empty:
                continue
            for tex, color in TEXTURE_COLORS.items():
                t = merged[merged["texture_condition"] == tex]
                if t.empty:
                    continue
                ax.scatter(
                    t["clip"],
                    t["dino"],
                    s=22,
                    alpha=0.7,
                    color=color,
                    edgecolor="black",
                    linewidth=0.4,
                    label=TEXTURE_DISPLAY[tex],
                )
            lo = min(merged["clip"].min(), merged["dino"].min(), -0.2)
            hi = max(merged["clip"].max(), merged["dino"].max(), 1.0)
            ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1)
            ax.axhline(0.0, color="black", linewidth=0.5)
            ax.axvline(0.0, color="black", linewidth=0.5)
            ax.set_xlabel(f"{MODEL_DISPLAY.get(clip_m, clip_m)} per-render Pearson r")
            ax.set_ylabel(f"{MODEL_DISPLAY.get(dino_m, dino_m)} per-render Pearson r")
            ax.set_title(f"layer = {layer}")
            ax.grid(linestyle=":", alpha=0.5)
            ax.legend(fontsize=8, loc="best")
            for _, row in merged.iterrows():
                rows_for_table.append(
                    dict(
                        layer=layer,
                        clip_model=clip_m,
                        dino_model=dino_m,
                        render_id=row["render_id"],
                        texture_condition=row["texture_condition"],
                        clip_pearson=row["clip"],
                        dino_pearson=row["dino"],
                        dino_minus_clip=row["dino"] - row["clip"],
                    )
                )
    fig.suptitle("Per-render Pearson r: DINOv2 ViT-B vs CLIP ViT-B/16 (object-only valid patches)")
    fig.tight_layout()
    save_figure(fig, out_dir / "dense_depth_pearson_scatter_clip_vs_dino.png")
    save_dataframe(pd.DataFrame(rows_for_table), out_dir / "dense_depth_pearson_scatter_clip_vs_dino.csv")
    plt.close(fig)


def plot_object_only_normals(object_only: pd.DataFrame, out_dir: Path) -> None:
    if object_only.empty:
        return
    df = object_only[object_only["task"] == "dense_surface_normal_patches"].copy()
    if df.empty:
        return
    plt = get_pyplot()
    layers = _ordered_layers(df)
    pairs = [("clip_vit_b16", "dinov2_vit_b")]
    fig, axes = plt.subplots(
        len(pairs), len(layers), figsize=(4.0 * len(layers) + 1.0, 4.0 * len(pairs)), squeeze=False
    )
    rows_for_table: List[dict] = []
    for r, (clip_m, dino_m) in enumerate(pairs):
        for c, layer in enumerate(layers):
            ax = axes[r, c]
            sub = df[df["layer"] == layer]
            clip_df = sub[sub["model"] == clip_m][
                ["render_id", "texture_condition", "angular_err_median_object"]
            ].rename(columns={"angular_err_median_object": "clip"})
            dino_df = sub[sub["model"] == dino_m][
                ["render_id", "texture_condition", "angular_err_median_object"]
            ].rename(columns={"angular_err_median_object": "dino"})
            merged = clip_df.merge(dino_df, on=["render_id", "texture_condition"], how="inner")
            if merged.empty:
                continue
            for tex, color in TEXTURE_COLORS.items():
                t = merged[merged["texture_condition"] == tex]
                if t.empty:
                    continue
                ax.scatter(
                    t["clip"],
                    t["dino"],
                    s=22,
                    alpha=0.7,
                    color=color,
                    edgecolor="black",
                    linewidth=0.4,
                    label=TEXTURE_DISPLAY[tex],
                )
            lo = 0
            hi = max(merged["clip"].max(), merged["dino"].max(), 90)
            ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1)
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_xlabel(f"{MODEL_DISPLAY.get(clip_m, clip_m)} per-render median angular error (deg)")
            ax.set_ylabel(f"{MODEL_DISPLAY.get(dino_m, dino_m)} per-render median angular error (deg)")
            ax.set_title(f"layer = {layer}")
            ax.grid(linestyle=":", alpha=0.5)
            ax.legend(fontsize=8, loc="best")
            for _, row in merged.iterrows():
                rows_for_table.append(
                    dict(
                        layer=layer,
                        render_id=row["render_id"],
                        texture_condition=row["texture_condition"],
                        clip_angerr_median=row["clip"],
                        dino_angerr_median=row["dino"],
                        clip_minus_dino=row["clip"] - row["dino"],
                    )
                )
    fig.suptitle("Per-render dense normals: CLIP vs DINO median angular error (object-only patches)")
    fig.tight_layout()
    save_figure(fig, out_dir / "dense_normals_angerr_scatter_clip_vs_dino.png")
    save_dataframe(pd.DataFrame(rows_for_table), out_dir / "dense_normals_angerr_scatter_clip_vs_dino.csv")
    plt.close(fig)


def write_reliability_summary(per_render: pd.DataFrame, out_dir: Path) -> None:
    summary = per_render_summary(per_render)
    save_dataframe(summary, out_dir / "dense_per_render_summary.csv")
    md = [
        "# Dense per-render reliability summary",
        "",
        "Per-render statistics computed from each dense probe's `predictions.csv`.",
        "Pearson r chance = 0; angular error chance ≈ 90°.",
        "",
        "## Dense depth (pearson_r)",
        "",
        "| model | layer | texture | n | mean | median | std | frac r > 0 | frac r > 0.2 | frac r > 0.4 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    depth = summary[summary["task"] == "dense_depth_patches"]
    for _, r in depth.iterrows():
        md.append(
            "| {m} | {l} | {t} | {n} | {mean:.3f} | {med:.3f} | {std:.3f} | {p0:.2f} | {p2:.2f} | {p4:.2f} |".format(
                m=r["model"], l=r["layer"], t=r["texture_condition"], n=int(r["n"]),
                mean=r["mean"], med=r["median"], std=r["std"],
                p0=r.get("frac_positive", 0.0),
                p2=r.get("frac_above_0_2", 0.0),
                p4=r.get("frac_above_0_4", 0.0),
            )
        )
    md += [
        "",
        "## Dense surface normals (per-render median angular error, deg)",
        "",
        "| model | layer | texture | n | mean | median | std | frac < 60° | frac < 45° | frac < 30° |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    normals = summary[summary["task"] == "dense_surface_normal_patches"]
    for _, r in normals.iterrows():
        md.append(
            "| {m} | {l} | {t} | {n} | {mean:.2f} | {med:.2f} | {std:.2f} | {p60:.2f} | {p45:.2f} | {p30:.2f} |".format(
                m=r["model"], l=r["layer"], t=r["texture_condition"], n=int(r["n"]),
                mean=r["mean"], med=r["median"], std=r["std"],
                p60=r.get("frac_below_60_deg", 0.0),
                p45=r.get("frac_below_45_deg", 0.0),
                p30=r.get("frac_below_30_deg", 0.0),
            )
        )
    (out_dir / "dense_per_render_summary.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_root = (args.dense_dir / "probes").resolve()
    print(f"[load] dense probes from {probe_root}")
    per_render = collect_dense_per_render(probe_root, args.split)
    print(f"[shape] per-render table: {per_render.shape}")

    plot_pearson_distributions(per_render, out_dir)
    print("[done] dense_depth_pearson_distribution")
    plot_pearson_ecdf(per_render, out_dir)
    print("[done] dense_depth_pearson_ecdf")
    plot_angular_error_distributions(per_render, out_dir)
    print("[done] dense_normals_angerr_distribution")

    models = sorted(per_render["model"].unique())
    layers = sorted(per_render["layer"].unique(), key=layer_sort_key)
    print(f"[recompute] object-only metrics from NPZ (models={models}, layers={layers})")
    obj_depth = recompute_per_render_dense_depth(probe_root, args.split, models, layers)
    obj_normals = recompute_per_render_dense_normals(probe_root, args.split, models, layers)
    object_only = pd.concat([obj_depth, obj_normals], ignore_index=True, sort=False)
    save_dataframe(object_only, out_dir / "dense_per_render_object_only.csv")

    plot_object_only_pearson(object_only, out_dir)
    print("[done] dense_depth_pearson_scatter_clip_vs_dino")
    plot_object_only_normals(object_only, out_dir)
    print("[done] dense_normals_angerr_scatter_clip_vs_dino")
    write_reliability_summary(per_render, out_dir)
    print("[done] dense_per_render_summary.csv/.md")

    print(f"[ok] figures written to: {out_dir}")


if __name__ == "__main__":
    main()

"""Render a grouped bar chart of dense-depth Pearson r by (model, layer, texture).

Reads outputs/exp1_dense_probe_analysis/figures/layerwise_densedepth.csv and
produces a clearer alternative to the existing line plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs/exp1_dense_probe_analysis/figures/layerwise_densedepth.csv"
OUT_DIR = ROOT / "outputs/exp1_dense_probe_analysis/figures"
BLOG_OUT = ROOT / "blog/experiment1/images"

MODEL_ORDER = ["clip_vit_b16", "dinov2_vit_b"]
MODEL_LABELS = {"clip_vit_b16": "CLIP ViT-B/16", "dinov2_vit_b": "DINOv2 ViT-B"}
LAYER_ORDER = ["layer8", "final"]
TEXTURE_ORDER = ["photorealistic", "flat", "random_noise"]
TEXTURE_LABELS = {
    "photorealistic": "Photorealistic",
    "flat": "Flat textureless",
    "random_noise": "Random noise",
}
TEXTURE_COLORS = {
    "photorealistic": "#1f77b4",
    "flat": "#7fb3d5",
    "random_noise": "#d1d5db",
}


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df[df["metric"] == "pearson_r_mean"].copy()

    fig, ax = plt.subplots(figsize=(9.0, 4.4))

    n_groups = len(MODEL_ORDER) * len(LAYER_ORDER)
    group_centers = np.arange(n_groups, dtype=float)
    n_tex = len(TEXTURE_ORDER)
    bar_w = 0.78 / n_tex

    xticklabels: list[str] = []
    for gi, (model, layer) in enumerate(
        [(m, l) for m in MODEL_ORDER for l in LAYER_ORDER]
    ):
        xticklabels.append(f"{MODEL_LABELS[model]}\n{layer}")
        for ti, tex in enumerate(TEXTURE_ORDER):
            row = df[
                (df["model_name"] == model)
                & (df["layer_name"] == layer)
                & (df["texture_condition"] == tex)
            ]
            if len(row) != 1:
                raise ValueError(f"Expected 1 row for {model}/{layer}/{tex}, got {len(row)}")
            val = float(row["value"].iloc[0])
            x = group_centers[gi] + (ti - (n_tex - 1) / 2) * bar_w
            ax.bar(
                x,
                val,
                width=bar_w,
                color=TEXTURE_COLORS[tex],
                edgecolor="#333333",
                linewidth=0.6,
                label=TEXTURE_LABELS[tex] if gi == 0 else None,
            )
            label_y = val + (0.012 if val >= 0 else -0.022)
            ax.text(
                x,
                label_y,
                f"{val:+.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8.2,
                color="#222222",
            )

    ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--", zorder=0)
    ax.text(
        n_groups - 0.5,
        0.005,
        "chance (r = 0)",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    for boundary in [1.5]:
        ax.axvline(boundary, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=0)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_ylabel("Per-render Pearson r (dense depth)  ↑", fontsize=10.5)
    ax.set_ylim(-0.10, 0.45)
    ax.set_title(
        "Dense depth: per-render Pearson r by model × ViT depth × texture",
        fontsize=11.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=9.5,
        handlelength=1.4,
        handleheight=1.0,
        ncol=3,
        bbox_to_anchor=(1.0, 1.0),
    )

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "layerwise_densedepth_bars.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")

    BLOG_OUT.mkdir(parents=True, exist_ok=True)
    blog_path = BLOG_OUT / "fig_dense_layerwise_bars.png"
    fig.savefig(blog_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out_path}")
    print(f"wrote {blog_path}")


if __name__ == "__main__":
    main()

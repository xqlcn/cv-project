#!/usr/bin/env python3
"""Write a compact diagnostic report for Experiment 1 outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.evaluation.diagnostics import (
    feature_cache_summary,
    manifest_integrity_summary,
    relative_depth_majority_baseline,
    relative_depth_pair_distribution,
)
from exp1.evaluation.metrics import load_results_table
from exp1.metadata.manifest import load_manifest


PRIMARY_METRICS = {
    "surface_normal_aggregate": ["angular_error_deg_mean", "angular_error_deg_median"],
    "relative_depth_regions": ["balanced_accuracy", "valid_pair_accuracy"],
}


def _resolve_required(project_root: Path, path: object) -> Path:
    resolved = resolve_path(project_root, str(path))
    assert resolved is not None
    return resolved


def _format_table(df: pd.DataFrame, *, max_rows: int = 80) -> str:
    if df.empty:
        return "_empty_\n"
    shown = df.head(max_rows).copy()
    try:
        return shown.to_markdown(index=False) + "\n"
    except Exception:
        return "```\n" + shown.to_string(index=False) + "\n```\n"


def _figure_inventory(figures_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(figures_dir.glob("*")):
        if path.suffix.lower() not in {".png", ".md"}:
            continue
        kind = "qualitative"
        if path.name.startswith("layerwise_"):
            kind = "layerwise"
        elif path.name.startswith("texture_drop_"):
            kind = "texture_drop"
        rows.append({"kind": kind, "count": 1})
    if not rows:
        return pd.DataFrame(columns=["kind", "count"])
    return pd.DataFrame(rows).groupby("kind", as_index=False)["count"].sum()


def _primary_result_tables(results: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    tables = []
    within = results[
        results["split"].astype(str).eq("test")
        & results["train_texture_condition"].astype(str).eq("all")
        & results["eval_texture_condition"].astype(str).eq("all")
    ].copy()
    for task, metrics in PRIMARY_METRICS.items():
        for metric in metrics:
            rows = within[
                within["task"].astype(str).eq(task)
                & within["metric"].astype(str).eq(metric)
            ].copy()
            if rows.empty:
                continue
            table = rows.pivot_table(
                index=["model", "layer"],
                columns="texture_condition",
                values="value",
                aggfunc="first",
            ).reset_index()
            tables.append((f"{task} / {metric}", table.round(4)))
    return tables


def _important_warnings(
    *,
    manifest: pd.DataFrame,
    rel_dist: Optional[pd.DataFrame],
) -> list[str]:
    warnings = []
    summary = manifest_integrity_summary(manifest)
    if summary.get("render_id_duplicates"):
        warnings.append("Duplicate render_id values are present.")
    if summary.get("object_split_leak_count"):
        warnings.append("At least one object_id appears in multiple splits.")
    category_count = int(manifest["category"].nunique()) if "category" in manifest else 0
    if category_count <= 1:
        warnings.append(
            "The run contains one category only, so category-level generalization "
            "claims are unsupported."
        )
    test_objects = summary.get("objects_by_split", {}).get("test", 0)
    if int(test_objects or 0) < 20:
        warnings.append(
            f"The test split has only {test_objects} object(s); object-level "
            "confidence intervals and rank orderings are unstable."
        )
    if rel_dist is not None and not rel_dist.empty:
        test_valid = rel_dist[rel_dist["split"].astype(str).eq("test")]
        active_pairs = int(test_valid[test_valid["n_valid"] > 0]["pair"].nunique())
        if active_pairs < 4:
            warnings.append(
                f"Relative depth uses only {active_pairs} active test pair(s); "
                "the nominal 3x3 task is effectively much narrower."
            )
    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    manifest_path = _resolve_required(project_root, cfg.paths.valid_render_manifest)
    labels_dir = _resolve_required(project_root, cfg.paths.labels_dir)
    feature_dir = _resolve_required(project_root, cfg.paths.feature_dir)
    results_dir = _resolve_required(project_root, cfg.paths.results_dir)
    figures_dir = _resolve_required(project_root, cfg.paths.figures_dir)
    results_path = args.results or (results_dir / "exp1_results_long.csv")
    output = args.output or (results_dir.parent / "FIGURE_DIAGNOSIS.md")
    if not output.is_absolute():
        output = project_root / output

    manifest = load_manifest(manifest_path, validate=False)
    results = load_results_table(results_path) if Path(results_path).is_file() else None
    surface_labels = labels_dir / "labels_surface_normal_aggregate.parquet"
    rel_labels = labels_dir / "labels_relative_depth_regions.parquet"

    rel_dist = None
    rel_baseline = None
    if rel_labels.is_file():
        rel_df = load_manifest(rel_labels, validate=False)
        rel_dist = relative_depth_pair_distribution(manifest, rel_df)
        rel_baseline = relative_depth_majority_baseline(manifest, rel_df)

    lines: list[str] = [
        "# Experiment 1 Figure Diagnosis",
        "",
        "This report is generated from the plotted figures, result tables, "
        "manifest, labels, and feature caches. It is intended as a sanity check "
        "before scientific interpretation.",
        "",
        "## Figure Inventory",
        "",
        _format_table(_figure_inventory(figures_dir)),
        "## Integrity Summary",
        "",
    ]
    summary = manifest_integrity_summary(manifest)
    lines.extend([f"- `{key}`: `{value}`" for key, value in summary.items()])
    warnings = _important_warnings(manifest=manifest, rel_dist=rel_dist)
    lines.extend(["", "## Automated Warnings", ""])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- No high-level integrity warnings were triggered.")

    lines.extend(["", "## Primary Test Results", ""])
    if results is not None:
        for title, table in _primary_result_tables(results):
            lines.extend([f"### {title}", "", _format_table(table)])
    else:
        lines.append("_No result table found._")

    lines.extend(["", "## Relative Depth Label Distribution", ""])
    if rel_dist is not None:
        compact = rel_dist[rel_dist["n_valid"] > 0].copy()
        lines.append(_format_table(compact.sort_values(["split", "pair"])))
    else:
        lines.append("_No relative-depth label table found._\n")

    lines.extend(["", "## Relative Depth Majority Baseline", ""])
    if rel_baseline is not None:
        lines.append(_format_table(rel_baseline.round(4)))
    else:
        lines.append("_No relative-depth label table found._\n")

    lines.extend(["", "## Feature Cache Alignment", ""])
    lines.append(_format_table(feature_cache_summary(feature_dir, manifest)))

    if surface_labels.is_file():
        surface = load_manifest(surface_labels, validate=False)
        lines.extend(["", "## Surface Normal Labels", ""])
        stats = surface[
            ["label_valid", "mean_normal_x", "mean_normal_y", "mean_normal_z"]
        ].describe(include="all")
        lines.append(_format_table(stats.reset_index().round(4)))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote diagnosis: {output}")


if __name__ == "__main__":
    main()

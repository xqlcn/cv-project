"""Comparison tables for Experiment 1 results."""

from __future__ import annotations

from typing import Optional

import pandas as pd

LOWER_IS_BETTER_TOKENS = ("error", "loss", "mae", "mse", "rmse", "median")
HIGHER_IS_BETTER_TOKENS = ("accuracy", "acc", "auc", "auroc", "f1", "r2")


def metric_direction(metric: str) -> Optional[str]:
    """Return whether a metric should increase or decrease when better."""
    text = str(metric).lower()
    if any(token in text for token in LOWER_IS_BETTER_TOKENS):
        return "lower"
    if any(token in text for token in HIGHER_IS_BETTER_TOKENS):
        return "higher"
    return None


def compute_texture_dependence_drops(
    results: pd.DataFrame,
    *,
    baseline_texture: str = "photorealistic",
) -> pd.DataFrame:
    """Compute performance degradation relative to a texture baseline.

    Positive ``texture_drop`` means the comparison texture is worse than the
    baseline according to the metric direction.
    """
    required = {
        "task",
        "model",
        "layer",
        "texture_condition",
        "split",
        "metric",
        "value",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError("Missing required results columns: " + ", ".join(missing))

    keys = ["task", "model", "layer", "split", "metric"]
    rows = []
    for key_values, group in results.groupby(keys, dropna=False):
        direction = metric_direction(str(key_values[-1]))
        if direction is None:
            continue
        baseline = group[group["texture_condition"].astype(str) == baseline_texture]
        if baseline.empty:
            continue
        baseline_value = float(baseline.iloc[0]["value"])
        for _, row in group.iterrows():
            texture = str(row["texture_condition"])
            if texture == baseline_texture:
                continue
            comparison_value = float(row["value"])
            raw_delta = comparison_value - baseline_value
            if direction == "lower":
                drop = raw_delta
            else:
                drop = baseline_value - comparison_value
            rows.append(
                {
                    "task": key_values[0],
                    "model": key_values[1],
                    "layer": key_values[2],
                    "split": key_values[3],
                    "metric": key_values[4],
                    "metric_direction": direction,
                    "baseline_texture": baseline_texture,
                    "comparison_texture": texture,
                    "baseline_value": baseline_value,
                    "comparison_value": comparison_value,
                    "raw_delta": raw_delta,
                    "texture_drop": float(drop),
                }
            )
    return pd.DataFrame(rows)

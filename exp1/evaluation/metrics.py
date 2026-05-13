"""Aggregate Experiment 1 probe metric artifacts into long tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from exp1.metadata.manifest import load_manifest, save_manifest


def _texture_label(value: Any) -> str:
    if value is None or value == "":
        return "all"
    if isinstance(value, (list, tuple)):
        return "+".join(str(item) for item in value) if value else "all"
    return str(value)


def _texture_eval_label(metadata: Mapping[str, Any]) -> str:
    texture = _texture_label(metadata.get("texture_condition"))
    train_texture = _texture_label(metadata.get("train_texture_condition"))
    eval_texture = _texture_label(metadata.get("eval_texture_condition"))
    if texture != "all":
        return texture
    if train_texture != "all" or eval_texture != "all":
        return f"train_{train_texture}__test_{eval_texture}"
    return "all"


def _path_metadata(path: Path, probe_root: Optional[Path]) -> dict[str, Any]:
    if probe_root is None:
        return {}
    try:
        parts = path.relative_to(probe_root).parts
    except ValueError:
        return {}
    if len(parts) < 4:
        return {}
    metadata: dict[str, Any] = {
        "model_name": parts[0],
        "layer_name": parts[1],
        "task": parts[2],
    }
    if len(parts) > 4 and parts[3].startswith("texture_"):
        metadata["texture_condition"] = parts[3].removeprefix("texture_")
    return metadata


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def load_probe_metrics_file(
    path: Union[str, Path],
    *,
    probe_root: Optional[Union[str, Path]] = None,
) -> list[dict[str, Any]]:
    """Load one ``metrics.json`` file as long metric records."""
    path = Path(path)
    root = Path(probe_root) if probe_root is not None else None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", payload)
    metadata = {
        **_path_metadata(path, root),
        **dict(payload.get("metadata", {})),
    }
    model_name = str(metadata.get("model_name", metadata.get("model", "unknown")))
    layer_name = str(metadata.get("layer_name", metadata.get("layer", "unknown")))
    task = str(metadata.get("task", "unknown"))
    texture_condition = _texture_eval_label(metadata)
    train_texture_condition = _texture_label(metadata.get("train_texture_condition"))
    eval_texture_condition = _texture_label(metadata.get("eval_texture_condition"))

    records: list[dict[str, Any]] = []
    for split, split_metrics in dict(metrics).items():
        if not isinstance(split_metrics, Mapping):
            continue
        for metric, value in split_metrics.items():
            numeric = _as_float(value)
            if numeric is None:
                continue
            records.append(
                {
                    "task": task,
                    "model": model_name,
                    "layer": layer_name,
                    "texture_condition": texture_condition,
                    "train_texture_condition": train_texture_condition,
                    "eval_texture_condition": eval_texture_condition,
                    "split": str(split),
                    "metric": str(metric),
                    "value": numeric,
                    "metrics_path": str(path),
                    "feature_mode": str(metadata.get("feature_mode", "")),
                    "target_mode": str(metadata.get("target_mode", "")),
                }
            )
    return records


def aggregate_probe_metrics(
    probe_root: Union[str, Path],
    *,
    pattern: str = "**/metrics.json",
) -> pd.DataFrame:
    """Aggregate every probe ``metrics.json`` under ``probe_root``."""
    root = Path(probe_root)
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            records.extend(load_probe_metrics_file(path, probe_root=root))
    columns = [
        "task",
        "model",
        "layer",
        "texture_condition",
        "train_texture_condition",
        "eval_texture_condition",
        "split",
        "metric",
        "value",
        "metrics_path",
        "feature_mode",
        "target_mode",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


BOOTSTRAP_PREDICTION_COLUMNS = {
    "angular_error_deg": "angular_error_deg_mean",
    "row_valid_pair_accuracy": "valid_pair_accuracy",
    "row_mae": "mae_mean",
    "viewpoint_angular_error_deg": "viewpoint_angular_error_deg_mean",
    "ssi_l1": "ssi_l1_mean",
    "scale_aware_abs_rel": "scale_aware_abs_rel_mean",
    "scale_aware_rmse": "scale_aware_rmse_mean",
    "scale_aware_rmse_log": "scale_aware_rmse_log_mean",
    "scale_aware_d1": "scale_aware_d1_mean",
    "scale_aware_d2": "scale_aware_d2_mean",
    "scale_aware_d3": "scale_aware_d3_mean",
    "scale_invariant_abs_rel": "scale_invariant_abs_rel_mean",
    "scale_invariant_rmse": "scale_invariant_rmse_mean",
    "scale_invariant_rmse_log": "scale_invariant_rmse_log_mean",
    "scale_invariant_d1": "scale_invariant_d1_mean",
    "scale_invariant_d2": "scale_invariant_d2_mean",
    "scale_invariant_d3": "scale_invariant_d3_mean",
    "pearson_r": "pearson_r_mean",
    "valid_patch_count": "valid_patch_count_mean",
}


def aggregate_prediction_bootstrap_cis(
    probe_root: Union[str, Path],
    *,
    pattern: str = "**/predictions.csv",
    unit_column: str = "object_id",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate object-level bootstrap CIs from saved prediction CSV files."""
    root = Path(probe_root)
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        metrics_path = path.with_name("metrics.json")
        metadata: dict[str, Any] = _path_metadata(path, root)
        if metrics_path.is_file():
            with metrics_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            metadata.update(dict(payload.get("metadata", {})))
        model_name = str(metadata.get("model_name", metadata.get("model", "unknown")))
        layer_name = str(metadata.get("layer_name", metadata.get("layer", "unknown")))
        task = str(metadata.get("task", "unknown"))
        texture_condition = _texture_eval_label(metadata)
        train_texture_condition = _texture_label(
            metadata.get("train_texture_condition")
        )
        eval_texture_condition = _texture_label(metadata.get("eval_texture_condition"))
        predictions = pd.read_csv(path)
        if "split" not in predictions.columns:
            continue
        for split, group in predictions.groupby("split", dropna=False):
            units = group[unit_column] if unit_column in group.columns else None
            for column, metric_name in BOOTSTRAP_PREDICTION_COLUMNS.items():
                if column not in group.columns:
                    continue
                values = pd.to_numeric(group[column], errors="coerce")
                valid = values.notna()
                if not valid.any():
                    continue
                ci = bootstrap_mean_ci(
                    values[valid].to_numpy(dtype=float),
                    units=(
                        units[valid].to_numpy()
                        if units is not None and len(units) == len(group)
                        else None
                    ),
                    n_resamples=int(n_resamples),
                    confidence=float(confidence),
                    seed=int(seed),
                )
                records.append(
                    {
                        "task": task,
                        "model": model_name,
                        "layer": layer_name,
                        "texture_condition": texture_condition,
                        "train_texture_condition": train_texture_condition,
                        "eval_texture_condition": eval_texture_condition,
                        "split": str(split),
                        "metric": metric_name,
                        "mean": ci["mean"],
                        "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"],
                        "feature_mode": str(metadata.get("feature_mode", "")),
                        "target_mode": str(metadata.get("target_mode", "")),
                        "unit_column": unit_column if units is not None else "",
                        "n_resamples": int(n_resamples),
                        "predictions_path": str(path),
                    }
                )
    columns = [
        "task",
        "model",
        "layer",
        "texture_condition",
        "train_texture_condition",
        "eval_texture_condition",
        "split",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "feature_mode",
        "target_mode",
        "unit_column",
        "n_resamples",
        "predictions_path",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def save_results_table(
    rows: Union[pd.DataFrame, Iterable[Mapping[str, Any]]],
    path: Union[str, Path],
) -> Path:
    """Save a long-format results table using existing manifest IO."""
    return save_manifest(rows, path, validate=False)


def load_results_table(path: Union[str, Path]) -> pd.DataFrame:
    """Load a previously aggregated Experiment 1 results table."""
    return load_manifest(path, validate=False)


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    units: Optional[Iterable[Any]] = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap a mean confidence interval, optionally resampling by object."""
    values_arr = np.asarray(list(values), dtype=np.float64)
    if values_arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(int(seed))
    alpha = (1.0 - float(confidence)) / 2.0
    if units is None:
        samples = [
            float(rng.choice(values_arr, size=values_arr.size, replace=True).mean())
            for _ in range(int(n_resamples))
        ]
    else:
        unit_arr = np.asarray([str(unit) for unit in units])
        if unit_arr.shape[0] != values_arr.shape[0]:
            raise ValueError("units and values must have the same length")
        unique_units = np.unique(unit_arr)
        samples = []
        for _ in range(int(n_resamples)):
            chosen = rng.choice(unique_units, size=unique_units.size, replace=True)
            sampled_values = np.concatenate(
                [values_arr[unit_arr == unit] for unit in chosen]
            )
            samples.append(float(sampled_values.mean()))

    return {
        "mean": float(values_arr.mean()),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
    }

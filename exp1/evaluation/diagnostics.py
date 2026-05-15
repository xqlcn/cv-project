"""Sanity checks for Experiment 1 result interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np
import pandas as pd


def _pair_indices(labels: pd.DataFrame) -> list[int]:
    indices = []
    for column in labels.columns:
        if column.startswith("pair_") and column.endswith("_label"):
            indices.append(int(column.split("_")[1]))
    return sorted(indices)


def _balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    recalls = []
    for cls in (0, 1):
        mask = labels == cls
        if mask.any():
            recalls.append(float((preds[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def manifest_integrity_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    """Return core split/control integrity statistics for a render manifest."""
    summary: dict[str, Any] = {
        "row_count": int(len(manifest)),
        "render_id_duplicates": int(manifest["render_id"].duplicated().sum())
        if "render_id" in manifest.columns
        else None,
        "object_count": int(manifest["object_id"].nunique())
        if "object_id" in manifest.columns
        else None,
        "split_counts": manifest.groupby("split").size().to_dict()
        if "split" in manifest.columns
        else {},
        "objects_by_split": manifest.groupby("split")["object_id"].nunique().to_dict()
        if {"split", "object_id"}.issubset(manifest.columns)
        else {},
    }
    if {"object_id", "split"}.issubset(manifest.columns):
        split_counts = manifest.groupby("object_id")["split"].nunique(dropna=False)
        summary["object_split_leak_count"] = int((split_counts > 1).sum())
    if "texture_control_group_id" in manifest.columns:
        groups = manifest.groupby("texture_control_group_id")
        summary["texture_control_group_count"] = int(groups.ngroups)
        summary["texture_control_bad_size_count"] = int((groups.size() != 3).sum())
        summary["texture_control_bad_texture_count"] = int(
            (groups["texture_condition"].nunique(dropna=False) != 3).sum()
        )
    if "qc_pass" in manifest.columns:
        summary["qc_pass_counts"] = manifest["qc_pass"].value_counts(
            dropna=False
        ).to_dict()
    return summary


def relative_depth_pair_distribution(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize valid relative-depth labels by split, texture, and pair."""
    joined = manifest.merge(labels, on="render_id", how="inner", validate="one_to_one")
    records = []
    for (split, texture), group in joined.groupby(["split", "texture_condition"]):
        for pair_idx in _pair_indices(labels):
            label_col = f"pair_{pair_idx}_label"
            valid_col = f"pair_{pair_idx}_valid"
            valid = group[valid_col].astype(bool)
            n_valid = int(valid.sum())
            n_pos = int(((group[label_col] == 1) & valid).sum())
            records.append(
                {
                    "split": str(split),
                    "texture_condition": str(texture),
                    "pair": int(pair_idx),
                    "n_valid": n_valid,
                    "n_positive": n_pos,
                    "positive_rate": float(n_pos / n_valid)
                    if n_valid
                    else float("nan"),
                }
            )
    return pd.DataFrame.from_records(records)


def relative_depth_majority_baseline(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    train_split: str = "train",
    eval_split: str = "test",
) -> pd.DataFrame:
    """Evaluate a train-set per-pair majority baseline on relative-depth labels."""
    joined = manifest.merge(labels, on="render_id", how="inner", validate="one_to_one")
    records = []
    for texture, texture_rows in joined.groupby("texture_condition"):
        train = texture_rows[texture_rows["split"].astype(str) == str(train_split)]
        eval_rows = texture_rows[texture_rows["split"].astype(str) == str(eval_split)]
        y_true_chunks = []
        train_majority_chunks = []
        oracle_majority_chunks = []
        eligible_pairs = 0
        for pair_idx in _pair_indices(labels):
            label_col = f"pair_{pair_idx}_label"
            valid_col = f"pair_{pair_idx}_valid"
            train_valid = train[valid_col].astype(bool)
            eval_valid = eval_rows[valid_col].astype(bool)
            if not train_valid.any() or not eval_valid.any():
                continue
            eligible_pairs += 1
            train_labels = train.loc[train_valid, label_col].astype(int)
            eval_labels = eval_rows.loc[eval_valid, label_col].astype(int)
            train_majority = int(train_labels.mean() >= 0.5)
            oracle_majority = int(eval_labels.mean() >= 0.5)
            y_true_chunks.append(eval_labels.to_numpy(dtype=np.int32))
            train_majority_chunks.append(
                np.full(len(eval_labels), train_majority, dtype=np.int32)
            )
            oracle_majority_chunks.append(
                np.full(len(eval_labels), oracle_majority, dtype=np.int32)
            )

        if y_true_chunks:
            y_true = np.concatenate(y_true_chunks)
            train_pred = np.concatenate(train_majority_chunks)
            oracle_pred = np.concatenate(oracle_majority_chunks)
            valid_count = int(y_true.shape[0])
            train_acc = float((y_true == train_pred).mean())
            oracle_acc = float((y_true == oracle_pred).mean())
            train_bal = _balanced_accuracy(y_true, train_pred)
            oracle_bal = _balanced_accuracy(y_true, oracle_pred)
        else:
            valid_count = 0
            train_acc = oracle_acc = train_bal = oracle_bal = float("nan")

        records.append(
            {
                "texture_condition": str(texture),
                "train_split": str(train_split),
                "eval_split": str(eval_split),
                "eligible_pair_count": int(eligible_pairs),
                "valid_pair_count": valid_count,
                "train_majority_valid_pair_accuracy": train_acc,
                "train_majority_balanced_accuracy": train_bal,
                "eval_oracle_majority_valid_pair_accuracy": oracle_acc,
                "eval_oracle_majority_balanced_accuracy": oracle_bal,
            }
        )
    return pd.DataFrame.from_records(records)


def feature_cache_summary(
    feature_dir: Union[str, Path],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Check global feature caches against manifest render IDs."""
    feature_root = Path(feature_dir)
    manifest_ids = set(manifest["render_id"].astype(str))
    records: list[Mapping[str, Any]] = []
    for path in sorted(feature_root.glob("*/*.npz")):
        if path.name.endswith("_patch.npz"):
            continue
        with np.load(path, allow_pickle=False) as data:
            if "render_ids" not in data or "features" not in data:
                continue
            render_ids = data["render_ids"].astype(str)
            features = data["features"]
        cache_ids = set(render_ids.tolist())
        records.append(
            {
                "cache_path": str(path),
                "row_count": int(len(render_ids)),
                "feature_dim": int(features.shape[1]) if features.ndim == 2 else None,
                "duplicate_render_ids": int(len(render_ids) - len(cache_ids)),
                "ids_missing_from_manifest": int(len(cache_ids - manifest_ids)),
                "manifest_ids_missing_from_cache": int(len(manifest_ids - cache_ids)),
            }
        )
    return pd.DataFrame.from_records(records)

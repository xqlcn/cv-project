"""Feature-cache datasets for Experiment 1 probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from exp1.data.label_join import join_labels_by_render_id
from exp1.data.splits import filter_manifest
from exp1.metadata.manifest import load_manifest


RowsLike = Union[pd.DataFrame, Iterable[Mapping[str, Any]]]


def _as_dataframe(rows_or_path: Union[RowsLike, str, Path]) -> pd.DataFrame:
    if isinstance(rows_or_path, (str, Path)):
        return load_manifest(rows_or_path, validate=False)
    if isinstance(rows_or_path, pd.DataFrame):
        return rows_or_path.copy()
    return pd.DataFrame(list(rows_or_path))


def load_feature_cache(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load an Experiment 1 npz feature cache with render_ids/features arrays."""
    with np.load(Path(path), allow_pickle=False) as data:
        if "render_ids" not in data or "features" not in data:
            raise KeyError("Feature cache must contain render_ids and features")
        return {
            "render_ids": data["render_ids"].astype(str),
            "features": data["features"].astype(np.float32),
        }


def relative_depth_columns(
    labels: pd.DataFrame,
) -> tuple[List[str], List[str]]:
    label_cols = sorted(
        [
            col
            for col in labels.columns
            if col.startswith("pair_") and col.endswith("_label")
        ],
        key=lambda col: int(col.split("_")[1]),
    )
    valid_cols = [col.replace("_label", "_valid") for col in label_cols]
    missing = [col for col in valid_cols if col not in labels.columns]
    if missing:
        raise KeyError("Missing relative-depth valid columns: " + ", ".join(missing))
    return label_cols, valid_cols


class Exp1FeatureDataset:
    """Load cached frozen features joined to task labels by render_id."""

    def __init__(
        self,
        feature_cache: Union[str, Path, Mapping[str, np.ndarray]],
        labels: Union[RowsLike, str, Path],
        *,
        manifest: Optional[Union[RowsLike, str, Path]] = None,
        task: str = "relative_depth_regions",
        target_columns: Optional[Sequence[str]] = None,
        split: Optional[Union[str, list[str]]] = None,
        texture_condition: Optional[Union[str, list[str]]] = None,
        require_label_valid: bool = True,
    ) -> None:
        cache = (
            load_feature_cache(feature_cache)
            if isinstance(feature_cache, (str, Path))
            else dict(feature_cache)
        )
        render_ids = np.asarray(cache["render_ids"]).astype(str)
        features = np.asarray(cache["features"], dtype=np.float32)
        if len(render_ids) != len(features):
            raise ValueError("render_ids and features have different lengths")
        if len(set(render_ids.tolist())) != len(render_ids):
            raise ValueError("Feature cache contains duplicate render_ids")

        feature_df = pd.DataFrame(
            {
                "render_id": render_ids,
                "_feature_index": np.arange(len(render_ids), dtype=np.int64),
            }
        )
        label_df = _as_dataframe(labels)
        joined = join_labels_by_render_id(feature_df, label_df, how="inner")
        if manifest is not None:
            manifest_df = _as_dataframe(manifest)
            joined = join_labels_by_render_id(manifest_df, joined, how="inner")
        joined = filter_manifest(
            joined,
            split=split,
            texture_condition=texture_condition,
            require_label_valid=require_label_valid,
        )

        self.features = features
        self.rows = joined.reset_index(drop=True)
        self.task = task
        if task == "relative_depth_regions":
            self.target_columns, self.valid_columns = relative_depth_columns(self.rows)
        else:
            if target_columns is None:
                target_columns = ("target",)
            self.target_columns = list(target_columns)
            self.valid_columns: List[str] = []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows.iloc[int(index)]
        feature = self.features[int(row["_feature_index"])]
        if self.task == "relative_depth_regions":
            target = row[self.target_columns].to_numpy(dtype=np.float32)
            valid_mask = row[self.valid_columns].to_numpy(dtype=bool)
        else:
            target = row[self.target_columns].to_numpy(dtype=np.float32)
            valid_mask = None
        return {
            "features": feature,
            "target": target,
            "valid_mask": valid_mask,
            "render_id": row["render_id"],
            "metadata": row.to_dict(),
        }

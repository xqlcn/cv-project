"""Render-manifest and label-table joins keyed by render_id."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Union

import pandas as pd

from exp1.metadata.manifest import load_manifest


RowsLike = Union[pd.DataFrame, Iterable[Mapping[str, Any]]]


class LabelJoinError(ValueError):
    """Raised when render_id joins are ambiguous or invalid."""


def _as_dataframe(rows: RowsLike) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(list(rows))


def _validate_render_ids(df: pd.DataFrame, *, name: str) -> None:
    if "render_id" not in df.columns:
        raise LabelJoinError(f"{name} is missing render_id")
    if df["render_id"].isna().any():
        raise LabelJoinError(f"{name} contains null render_id values")
    duplicates = df["render_id"].duplicated(keep=False)
    if duplicates.any():
        values = sorted(str(v) for v in df.loc[duplicates, "render_id"].unique())
        raise LabelJoinError(f"{name} has duplicate render_id values: {values[:8]}")


def join_labels_by_render_id(
    manifest_rows: RowsLike,
    label_rows: RowsLike,
    *,
    how: str = "inner",
    validate: bool = True,
) -> pd.DataFrame:
    """Join manifest rows to labels by render_id, never by row order."""
    manifest = _as_dataframe(manifest_rows)
    labels = _as_dataframe(label_rows)
    if validate:
        _validate_render_ids(manifest, name="manifest")
        _validate_render_ids(labels, name="labels")
    joined = manifest.merge(
        labels,
        on="render_id",
        how=how,
        validate="one_to_one" if validate else None,
        suffixes=("", "_label"),
    )
    return joined.reset_index(drop=True)


def load_and_join_labels(
    manifest_path: Union[str, Path],
    label_path: Union[str, Path],
    *,
    how: str = "inner",
) -> pd.DataFrame:
    """Load manifest/label files and join them by render_id."""
    manifest = load_manifest(manifest_path, validate=False)
    labels = load_manifest(label_path, validate=False)
    return join_labels_by_render_id(manifest, labels, how=how)

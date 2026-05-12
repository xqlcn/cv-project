"""Split filtering and leakage checks for Experiment 1."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd


RowsLike = Union[pd.DataFrame, Iterable[Mapping[str, Any]]]


class SplitValidationError(ValueError):
    """Raised when object-disjoint split assumptions are violated."""


def _as_dataframe(rows: RowsLike) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(list(rows))


def assert_object_disjoint_splits(
    rows: RowsLike,
    *,
    object_column: str = "object_id",
    split_column: str = "split",
) -> pd.DataFrame:
    """Assert each object_id appears in at most one split."""
    df = _as_dataframe(rows)
    for column in (object_column, split_column):
        if column not in df.columns:
            raise SplitValidationError(f"Missing required split column: {column}")

    split_counts = df.groupby(object_column)[split_column].nunique(dropna=False)
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        preview = ", ".join(str(v) for v in leaked.index[:8])
        raise SplitValidationError(
            "Objects appear in multiple splits: " + preview
        )
    return df


def filter_manifest(
    rows: RowsLike,
    *,
    split: Optional[Union[str, Sequence[str]]] = None,
    texture_condition: Optional[Union[str, Sequence[str]]] = None,
    require_qc_pass: bool = False,
    require_label_valid: bool = False,
) -> pd.DataFrame:
    """Filter manifest-like rows by split, texture, QC, and label validity."""
    df = _as_dataframe(rows)
    if split is not None:
        allowed = {str(split)} if isinstance(split, str) else {str(v) for v in split}
        df = df[df["split"].astype(str).isin(allowed)]
    if texture_condition is not None:
        allowed = (
            {str(texture_condition)}
            if isinstance(texture_condition, str)
            else {str(v) for v in texture_condition}
        )
        df = df[df["texture_condition"].astype(str).isin(allowed)]
    if require_qc_pass and "qc_pass" in df.columns:
        df = df[df["qc_pass"].astype(bool)]
    if require_label_valid and "label_valid" in df.columns:
        df = df[df["label_valid"].astype(bool)]
    return df.reset_index(drop=True)

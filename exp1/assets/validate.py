"""Validation helpers for Experiment 1 asset manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import pandas as pd

from src.utils.io import write_jsonl


REQUIRED_ASSET_COLUMNS = (
    "object_id",
    "source_dataset",
    "category",
    "split",
    "raw_mesh_path",
    "normalized_mesh_path",
    "asset_status",
    "asset_error_message",
)
VALID_ASSET_STATUSES = {"discovered", "normalized", "failed"}
VALID_SPLITS = {"train", "val", "test"}


class AssetValidationError(ValueError):
    """Raised when an asset manifest violates the Experiment 1 contract."""


def _as_dataframe(
    rows: Union[pd.DataFrame, Iterable[Mapping[str, Any]]],
) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(list(rows))


def _missing_columns(df: pd.DataFrame, required: Sequence[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def assert_object_disjoint_splits(
    rows: Union[pd.DataFrame, Iterable[Mapping[str, Any]]],
) -> None:
    df = _as_dataframe(rows)
    missing = _missing_columns(df, ("object_id", "split"))
    if missing:
        raise AssetValidationError("Missing split columns: " + ", ".join(missing))
    grouped = df.groupby("object_id")["split"].nunique()
    leaking = grouped[grouped > 1]
    if not leaking.empty:
        examples = ", ".join(str(idx) for idx in leaking.index[:8])
        raise AssetValidationError(f"Objects assigned to multiple splits: {examples}")


def validate_asset_manifest(
    rows: Union[pd.DataFrame, Iterable[Mapping[str, Any]]],
    *,
    require_normalized_paths: bool = True,
) -> pd.DataFrame:
    """Validate path and metadata fields for asset manifests."""
    df = _as_dataframe(rows)
    missing = _missing_columns(df, REQUIRED_ASSET_COLUMNS)
    if missing:
        raise AssetValidationError("Missing asset columns: " + ", ".join(missing))

    invalid_splits = sorted(set(df["split"]) - VALID_SPLITS)
    if invalid_splits:
        raise AssetValidationError(f"Invalid split values: {invalid_splits}")

    invalid_statuses = sorted(set(df["asset_status"]) - VALID_ASSET_STATUSES)
    if invalid_statuses:
        raise AssetValidationError(f"Invalid asset statuses: {invalid_statuses}")

    usable = df["asset_status"] != "failed"
    missing_raw = [
        str(p)
        for p in df.loc[usable, "raw_mesh_path"]
        if not str(p) or not Path(str(p)).is_file()
    ]
    if missing_raw:
        raise AssetValidationError(f"Missing raw mesh paths: {missing_raw[:8]}")

    if require_normalized_paths:
        ok = df["asset_status"] == "normalized"
        missing_normalized = [
            str(p)
            for p in df.loc[ok, "normalized_mesh_path"]
            if not str(p) or not Path(str(p)).is_file()
        ]
        if missing_normalized:
            raise AssetValidationError(
                f"Missing normalized mesh paths: {missing_normalized[:8]}"
            )

    assert_object_disjoint_splits(df)
    return df


def build_object_split_manifest(
    rows: Union[pd.DataFrame, Iterable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Build one split-assignment row per object_id."""
    df = _as_dataframe(rows)
    assert_object_disjoint_splits(df)
    required = ("object_id", "source_dataset", "category", "split")
    missing = _missing_columns(df, required)
    if missing:
        raise AssetValidationError("Missing split columns: " + ", ".join(missing))

    split_df = (
        df.loc[:, list(required)]
        .drop_duplicates(subset=["object_id"])
        .sort_values(["split", "category", "object_id"])
    )
    return split_df.to_dict(orient="records")


def write_object_split_manifest(
    rows: Union[pd.DataFrame, Iterable[Mapping[str, Any]]],
    path: Union[str, Path],
) -> Path:
    split_rows = build_object_split_manifest(rows)
    path = Path(path)
    write_jsonl(path, split_rows)
    return path

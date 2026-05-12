"""Validation helpers for Experiment 1 asset manifests."""

from __future__ import annotations

import hashlib
import json
import math
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
DEFAULT_SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


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


def _stable_object_key(row: Mapping[str, Any], *, seed: int) -> str:
    payload = {
        "seed": int(seed),
        "object_id": str(row.get("object_id", "")),
        "source_dataset": str(row.get("source_dataset", "")),
        "category": str(row.get("category", "")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _normalized_fractions(
    fractions: Mapping[str, Any],
    labels: Sequence[str],
) -> list[float]:
    values = []
    for label in labels:
        value = float(fractions.get(label, 0.0))
        if value < 0:
            raise AssetValidationError(f"Split fraction for {label!r} is negative")
        values.append(value)
    total = float(sum(values))
    if total <= 0:
        raise AssetValidationError("Split fractions must sum to a positive value")
    return [value / total for value in values]


def _split_counts(
    n_objects: int,
    *,
    labels: Sequence[str],
    fractions: Mapping[str, Any],
) -> list[int]:
    if n_objects <= 0:
        return [0 for _ in labels]
    normalized = _normalized_fractions(fractions, labels)
    raw = [n_objects * fraction for fraction in normalized]
    counts = [int(math.floor(value)) for value in raw]
    remainder = n_objects - sum(counts)
    order = sorted(
        range(len(labels)),
        key=lambda idx: (raw[idx] - counts[idx], normalized[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    positive = [idx for idx, fraction in enumerate(normalized) if fraction > 0]
    if n_objects >= len(positive):
        for idx in positive:
            if counts[idx] > 0:
                continue
            donors = [donor for donor in positive if donor != idx and counts[donor] > 1]
            if not donors:
                continue
            donor = max(
                donors,
                key=lambda donor_idx: (
                    counts[donor_idx] - raw[donor_idx],
                    counts[donor_idx],
                ),
            )
            counts[donor] -= 1
            counts[idx] += 1
    return counts


def assign_object_disjoint_splits(
    rows: Union[pd.DataFrame, Iterable[Mapping[str, Any]]],
    *,
    fractions: Mapping[str, Any] = DEFAULT_SPLIT_FRACTIONS,
    labels: Sequence[str] = ("train", "val", "test"),
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Assign deterministic object-disjoint splits from configured fractions.

    The assignment is based on one stable hash per object and is then joined
    back to every row by ``object_id``. This keeps all renders/meshes for an
    object in one split even when the input manifest has duplicate rows.
    """
    df = _as_dataframe(rows)
    missing = _missing_columns(df, ("object_id", "source_dataset", "category"))
    if missing:
        raise AssetValidationError(
            "Missing split assignment columns: " + ", ".join(missing)
        )

    labels = [str(label) for label in labels]
    invalid_labels = sorted(set(labels) - VALID_SPLITS)
    if invalid_labels:
        raise AssetValidationError(f"Invalid split labels: {invalid_labels}")

    object_rows = (
        df.loc[:, ["object_id", "source_dataset", "category"]]
        .drop_duplicates(subset=["object_id"])
        .to_dict(orient="records")
    )
    object_rows = sorted(
        object_rows,
        key=lambda row: _stable_object_key(row, seed=int(seed)),
    )
    counts = _split_counts(
        len(object_rows),
        labels=labels,
        fractions=fractions,
    )

    assignments: dict[str, str] = {}
    cursor = 0
    for label, count in zip(labels, counts):
        for row in object_rows[cursor : cursor + count]:
            assignments[str(row["object_id"])] = label
        cursor += count

    out = df.copy()
    if "split" in out.columns:
        out["split_original"] = out["split"].astype(str)
    out["split"] = out["object_id"].astype(str).map(assignments)
    out["split_assignment_method"] = "deterministic_fraction"
    out["split_assignment_seed"] = int(seed)
    assert_object_disjoint_splits(out)
    return out.to_dict(orient="records")


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

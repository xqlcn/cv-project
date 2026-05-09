"""Load, save, and validate Experiment 1 manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import pandas as pd

from exp1.metadata.schema import (
    REQUIRED_RENDER_COLUMNS,
    VALID_SPLITS,
    VALID_TEXTURE_CONDITIONS,
    generate_render_id,
)
from src.utils.io import read_jsonl, write_jsonl


RowsLike = Union[pd.DataFrame, Iterable[Mapping[str, Any]]]


class ManifestValidationError(ValueError):
    """Raised when a manifest violates the Experiment 1 schema contract."""


def _as_dataframe(rows: RowsLike) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(list(rows))


def _preview(values: Sequence[Any], *, limit: int = 8) -> str:
    shown = [repr(v) for v in values[:limit]]
    if len(values) > limit:
        shown.append(f"... +{len(values) - limit} more")
    return ", ".join(shown)


def validate_required_columns(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str] = REQUIRED_RENDER_COLUMNS,
) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ManifestValidationError(
            "Missing required manifest columns: " + ", ".join(missing)
        )


def validate_split_values(
    df: pd.DataFrame,
    *,
    column: str = "split",
    allowed: Sequence[str] = VALID_SPLITS,
) -> None:
    if column not in df.columns:
        raise ManifestValidationError(f"Missing required manifest column: {column}")
    if df[column].isna().any():
        raise ManifestValidationError(
            f"Manifest column '{column}' contains null values"
        )
    invalid = sorted(
        {str(v) for v in df[column].dropna().unique() if str(v) not in set(allowed)}
    )
    if invalid:
        raise ManifestValidationError(
            f"Invalid split values in column '{column}': {_preview(invalid)}. "
            f"Allowed values: {', '.join(allowed)}"
        )


def validate_texture_conditions(
    df: pd.DataFrame,
    *,
    column: str = "texture_condition",
    allowed: Sequence[str] = VALID_TEXTURE_CONDITIONS,
) -> None:
    if column not in df.columns:
        raise ManifestValidationError(f"Missing required manifest column: {column}")
    if df[column].isna().any():
        raise ManifestValidationError(
            f"Manifest column '{column}' contains null values"
        )
    allowed_set = set(allowed)
    invalid = sorted(
        {str(v) for v in df[column].dropna().unique() if str(v) not in allowed_set}
    )
    if invalid:
        raise ManifestValidationError(
            f"Invalid texture conditions in column '{column}': {_preview(invalid)}. "
            f"Allowed values: {', '.join(allowed)}"
        )


def validate_duplicate_ids(df: pd.DataFrame, *, id_column: str = "render_id") -> None:
    if id_column not in df.columns:
        raise ManifestValidationError(f"Missing required manifest column: {id_column}")
    if df[id_column].isna().any():
        raise ManifestValidationError(
            f"Manifest column '{id_column}' contains null IDs"
        )

    duplicate_mask = df[id_column].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = sorted(
            str(v) for v in df.loc[duplicate_mask, id_column].unique()
        )
        raise ManifestValidationError(
            f"Duplicate {id_column} values: {_preview(duplicate_ids)}"
        )


def validate_render_manifest(
    rows: RowsLike,
    *,
    required_columns: Sequence[str] = REQUIRED_RENDER_COLUMNS,
    allowed_splits: Sequence[str] = VALID_SPLITS,
    allowed_texture_conditions: Sequence[str] = VALID_TEXTURE_CONDITIONS,
) -> pd.DataFrame:
    """Validate an Experiment 1 render manifest and return a DataFrame copy."""
    df = _as_dataframe(rows)
    validate_required_columns(df, required_columns=required_columns)
    validate_split_values(df, allowed=allowed_splits)
    validate_texture_conditions(df, allowed=allowed_texture_conditions)
    validate_duplicate_ids(df)
    return df


def attach_render_ids(
    rows: RowsLike,
    *,
    overwrite: bool = False,
    id_column: str = "render_id",
) -> pd.DataFrame:
    """Return a DataFrame with deterministic render IDs attached."""
    df = _as_dataframe(rows)
    if id_column in df.columns and not overwrite:
        missing_mask = df[id_column].isna() | (df[id_column].astype(str) == "")
        if not missing_mask.any():
            return df
        for idx, record in df.loc[missing_mask].to_dict(orient="index").items():
            df.at[idx, id_column] = generate_render_id(record)
        return df

    ids = [generate_render_id(record) for record in df.to_dict(orient="records")]
    df[id_column] = ids
    return df


def load_manifest(
    path: Union[str, Path],
    *,
    validate: bool = True,
    required_columns: Sequence[str] = REQUIRED_RENDER_COLUMNS,
) -> pd.DataFrame:
    """Load a manifest from JSONL, JSON, CSV, or Parquet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        df = pd.DataFrame(read_jsonl(path))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "rows" in payload:
            payload = payload["rows"]
        df = pd.DataFrame(payload)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet manifests requires pyarrow or fastparquet. "
                "Use .jsonl/.csv or install a Parquet engine."
            ) from exc
    else:
        raise ValueError(
            f"Unsupported manifest extension '{path.suffix}'. "
            "Use .jsonl, .json, .csv, or .parquet."
        )

    if validate:
        return validate_render_manifest(df, required_columns=required_columns)
    return df


def save_manifest(
    rows: RowsLike,
    path: Union[str, Path],
    *,
    validate: bool = True,
    required_columns: Sequence[str] = REQUIRED_RENDER_COLUMNS,
) -> Path:
    """Save a manifest to JSONL, JSON, CSV, or Parquet."""
    path = Path(path)
    df = _as_dataframe(rows)
    if validate:
        df = validate_render_manifest(df, required_columns=required_columns)

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        write_jsonl(path, df.to_dict(orient="records"))
    elif suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)
            f.write("\n")
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Writing Parquet manifests requires pyarrow or fastparquet. "
                "Use .jsonl/.csv or install a Parquet engine."
            ) from exc
    else:
        raise ValueError(
            f"Unsupported manifest extension '{path.suffix}'. "
            "Use .jsonl, .json, .csv, or .parquet."
        )
    return path

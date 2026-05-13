"""Relative-depth region-pair labels for Experiment 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


DEFAULT_REGION_PAIRS = (
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (0, 3),
    (3, 6),
    (1, 4),
    (4, 7),
    (2, 5),
    (5, 8),
)


def _resolve_path(project_root: Optional[Union[str, Path]], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


def _region_slices(
    shape: tuple[int, int],
    grid_size: Sequence[int],
) -> List[tuple[slice, slice]]:
    height, width = shape
    rows, cols = [int(v) for v in grid_size]
    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, cols + 1, dtype=int)
    slices: List[tuple[slice, slice]] = []
    for row in range(rows):
        for col in range(cols):
            slices.append(
                (
                    slice(int(y_edges[row]), int(y_edges[row + 1])),
                    slice(int(x_edges[col]), int(x_edges[col + 1])),
                )
            )
    return slices


def _foreground_bbox(
    mask: np.ndarray,
    *,
    padding_fraction: float = 0.0,
) -> Optional[tuple[slice, slice]]:
    ys, xs = np.where(mask.astype(bool))
    if len(ys) == 0 or len(xs) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = int(np.ceil((y1 - y0) * float(padding_fraction)))
    pad_x = int(np.ceil((x1 - x0) * float(padding_fraction)))
    y0 = max(0, y0 - pad_y)
    y1 = min(mask.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(mask.shape[1], x1 + pad_x)
    return slice(y0, y1), slice(x0, x1)


def _depth_stat(values: np.ndarray, statistic: str) -> float:
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    raise ValueError(f"Unsupported depth statistic: {statistic}")


def region_depths(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    grid_size: Sequence[int] = (3, 3),
    statistic: str = "median",
    min_valid_fraction_per_region: float = 0.25,
    use_foreground_bbox: bool = False,
    bbox_padding_fraction: float = 0.0,
) -> tuple[List[float], List[bool], List[int]]:
    """Return one depth statistic and valid flag per coarse image region."""
    if depth.ndim != 2:
        raise ValueError(f"Expected depth [H,W], got shape {depth.shape}")
    if mask.shape != depth.shape:
        raise ValueError("Depth and mask shapes differ")
    if use_foreground_bbox:
        bbox = _foreground_bbox(mask, padding_fraction=bbox_padding_fraction)
        if bbox is not None:
            ys, xs = bbox
            depth = depth[ys, xs]
            mask = mask[ys, xs]

    depths: List[float] = []
    valid: List[bool] = []
    counts: List[int] = []
    for ys, xs in _region_slices(depth.shape, grid_size):
        region_depth = depth[ys, xs]
        region_mask = mask[ys, xs].astype(bool)
        finite = region_mask & np.isfinite(region_depth)
        min_count = max(1, int(np.ceil(finite.size * min_valid_fraction_per_region)))
        count = int(finite.sum())
        counts.append(count)
        if count < min_count:
            depths.append(np.nan)
            valid.append(False)
            continue
        depths.append(_depth_stat(region_depth[finite], statistic))
        valid.append(True)
    return depths, valid, counts


def relative_depth_label_row(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    region_pairs: Sequence[Sequence[int]] = DEFAULT_REGION_PAIRS,
    grid_size: Sequence[int] = (3, 3),
    statistic: str = "median",
    min_valid_fraction_per_region: float = 0.25,
    min_depth_margin: float = 0.02,
    use_foreground_bbox: bool = False,
    bbox_padding_fraction: float = 0.0,
) -> Dict[str, Any]:
    """Create wide relative-depth pair labels for one render."""
    depths, region_valid, counts = region_depths(
        depth,
        mask,
        grid_size=grid_size,
        statistic=statistic,
        min_valid_fraction_per_region=min_valid_fraction_per_region,
        use_foreground_bbox=use_foreground_bbox,
        bbox_padding_fraction=bbox_padding_fraction,
    )

    row: Dict[str, Any] = {
        "relative_depth_grid_rows": int(grid_size[0]),
        "relative_depth_grid_cols": int(grid_size[1]),
        "relative_depth_pair_count": len(region_pairs),
        "relative_depth_region_frame": (
            "foreground_bbox" if use_foreground_bbox else "full_image"
        ),
        "relative_depth_bbox_padding_fraction": float(bbox_padding_fraction),
    }
    for idx, value in enumerate(depths):
        row[f"region_{idx}_depth"] = float(value)
        row[f"region_{idx}_valid"] = bool(region_valid[idx])
        row[f"region_{idx}_valid_pixel_count"] = int(counts[idx])

    valid_pair_count = 0
    for pair_idx, pair in enumerate(region_pairs):
        a, b = [int(v) for v in pair]
        valid = bool(region_valid[a] and region_valid[b])
        label = -1
        if valid:
            diff = float(depths[a] - depths[b])
            valid = abs(diff) >= float(min_depth_margin)
            if valid:
                label = int(depths[a] < depths[b])
                valid_pair_count += 1
        row[f"pair_{pair_idx}_region_a"] = a
        row[f"pair_{pair_idx}_region_b"] = b
        row[f"pair_{pair_idx}_label"] = label
        row[f"pair_{pair_idx}_valid"] = bool(valid)

    row["relative_depth_valid_pair_count"] = int(valid_pair_count)
    row["label_valid"] = valid_pair_count > 0
    row["label_error_message"] = "" if valid_pair_count > 0 else "no_valid_pairs"
    return row


def build_relative_depth_labels(
    rows: Iterable[Mapping[str, Any]],
    *,
    project_root: Optional[Union[str, Path]] = None,
    region_pairs: Sequence[Sequence[int]] = DEFAULT_REGION_PAIRS,
    grid_size: Sequence[int] = (3, 3),
    statistic: str = "median",
    min_valid_fraction_per_region: float = 0.25,
    min_depth_margin: float = 0.02,
    use_foreground_bbox: bool = False,
    bbox_padding_fraction: float = 0.0,
) -> pd.DataFrame:
    """Build wide relative-depth labels for render manifest rows."""
    labels = []
    for manifest_row in rows:
        out: Dict[str, Any] = {"render_id": str(manifest_row["render_id"])}
        try:
            depth = np.load(
                _resolve_path(project_root, manifest_row["depth_path"]),
                allow_pickle=False,
            )
            mask = np.load(
                _resolve_path(project_root, manifest_row["mask_path"]),
                allow_pickle=False,
            )
            out.update(
                relative_depth_label_row(
                    depth,
                    mask,
                    region_pairs=region_pairs,
                    grid_size=grid_size,
                    statistic=statistic,
                    min_valid_fraction_per_region=min_valid_fraction_per_region,
                    min_depth_margin=min_depth_margin,
                    use_foreground_bbox=use_foreground_bbox,
                    bbox_padding_fraction=bbox_padding_fraction,
                )
            )
        except Exception as exc:
            out.update(
                {
                    "relative_depth_pair_count": len(region_pairs),
                    "relative_depth_valid_pair_count": 0,
                    "label_valid": False,
                    "label_error_message": f"{type(exc).__name__}:{exc}",
                }
            )
        labels.append(out)
    return pd.DataFrame(labels)


def relative_depth_coverage_summary(
    manifest_rows: Iterable[Mapping[str, Any]],
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize active pair coverage by split and texture condition."""
    manifest = pd.DataFrame(list(manifest_rows))
    if manifest.empty or labels.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "texture_condition",
                "example_count",
                "valid_example_count",
                "active_pair_count",
            ]
        )
    required = {"render_id", "split", "texture_condition"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError("Manifest missing relative-depth coverage columns: " + ", ".join(missing))
    merged = manifest.loc[:, ["render_id", "split", "texture_condition"]].merge(
        labels,
        on="render_id",
        how="inner",
        validate="one_to_one",
    )
    pair_valid_cols = sorted(
        [
            col
            for col in labels.columns
            if col.startswith("pair_") and col.endswith("_valid")
        ],
        key=lambda name: int(name.split("_")[1]),
    )
    rows = []
    for (split, texture), group in merged.groupby(["split", "texture_condition"]):
        active_pairs = int(sum(bool(group[col].astype(bool).any()) for col in pair_valid_cols))
        valid_examples = (
            int(group["label_valid"].astype(bool).sum())
            if "label_valid" in group.columns
            else 0
        )
        rows.append(
            {
                "split": str(split),
                "texture_condition": str(texture),
                "example_count": int(len(group)),
                "valid_example_count": valid_examples,
                "active_pair_count": active_pairs,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "texture_condition"]).reset_index(
        drop=True
    )


def validate_relative_depth_coverage(
    manifest_rows: Iterable[Mapping[str, Any]],
    labels: pd.DataFrame,
    *,
    min_active_pairs: int = 4,
    min_valid_examples: int = 1,
) -> pd.DataFrame:
    """Raise when any split/texture has insufficient relative-depth coverage."""
    summary = relative_depth_coverage_summary(manifest_rows, labels)
    if summary.empty:
        raise ValueError("Relative-depth coverage is empty")
    bad = summary[
        (summary["active_pair_count"] < int(min_active_pairs))
        | (summary["valid_example_count"] < int(min_valid_examples))
    ]
    if not bad.empty:
        raise ValueError(
            "Insufficient relative-depth label coverage:\n"
            + bad.to_string(index=False)
        )
    return summary

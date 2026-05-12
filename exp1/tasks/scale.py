"""Object scale and apparent-size labels for Experiment 1."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd


def _resolve_path(project_root: Optional[Union[str, Path]], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


def foreground_area_fraction(
    row: Mapping[str, Any],
    *,
    project_root: Optional[Union[str, Path]] = None,
) -> float:
    """Return foreground mask fraction for one render row."""
    if row.get("qc_foreground_fraction") not in {None, ""}:
        return float(row["qc_foreground_fraction"])
    mask = np.load(_resolve_path(project_root, row["mask_path"]), allow_pickle=False)
    return float(mask.astype(bool).mean())


def build_scale_labels(
    rows: Iterable[Mapping[str, Any]],
    *,
    project_root: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build object-scale/apparent-size labels keyed by render_id."""
    labels = []
    for row in rows:
        scale = float(row["object_scale"])
        out = {
            "render_id": str(row["render_id"]),
            "object_scale": scale,
            "log_object_scale": math.log(max(scale, 1e-8)),
            "label_valid": True,
            "label_error_message": "",
        }
        try:
            out["foreground_area_fraction"] = foreground_area_fraction(
                row,
                project_root=project_root,
            )
        except Exception as exc:
            out["foreground_area_fraction"] = np.nan
            out["label_valid"] = False
            out["label_error_message"] = f"{type(exc).__name__}:{exc}"
        labels.append(out)
    return pd.DataFrame(labels)

"""Schema constants and stable identifiers for Experiment 1 manifests."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


VALID_SPLITS = ("train", "val", "test")
VALID_TEXTURE_CONDITIONS = ("photorealistic", "flat", "random_noise")

REQUIRED_RENDER_COLUMNS = (
    "render_id",
    "object_id",
    "source_dataset",
    "category",
    "split",
    "raw_mesh_path",
    "normalized_mesh_path",
    "texture_condition",
    "texture_seed",
    "camera_distance",
    "camera_azimuth_deg",
    "camera_elevation_deg",
    "camera_fov_deg",
    "object_scale",
    "light_type",
    "light_azimuth_deg",
    "light_elevation_deg",
    "light_intensity",
    "rgb_path",
    "depth_path",
    "normal_path",
    "mask_path",
    "render_seed",
    "render_status",
    "qc_error_message",
)

# Fields that define one controlled render, excluding output paths and mutable
# status/QC columns so IDs are stable across machines and reruns.
RENDER_ID_FIELDS = (
    "object_id",
    "source_dataset",
    "texture_condition",
    "texture_seed",
    "camera_distance",
    "camera_azimuth_deg",
    "camera_elevation_deg",
    "camera_fov_deg",
    "object_scale",
    "light_type",
    "light_azimuth_deg",
    "light_elevation_deg",
    "light_intensity",
    "render_seed",
)


def _canonical_value(value: Any) -> Any:
    """Convert values to a JSON-stable representation for hashing."""
    if value is None:
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric_value = float(value)
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            raise ValueError(
                f"Cannot generate render_id from non-finite float: {value}"
            )
        return format(numeric_value, ".12g")
    if isinstance(value, (list, tuple)):
        return [_canonical_value(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _canonical_value(value[k]) for k in sorted(value)}
    return str(value)


def generate_render_id(
    record: Mapping[str, Any],
    *,
    fields: Sequence[str] = RENDER_ID_FIELDS,
    prefix: str = "r",
    digest_size: int = 16,
    version: Optional[str] = "v1",
) -> str:
    """Generate a deterministic render ID from the controlled render fields.

    The hash deliberately excludes output file paths, render status, and QC
    fields. Moving the dataset or re-validating a render should not change its
    identity.
    """
    missing = [field for field in fields if field not in record]
    if missing:
        raise KeyError(
            "Cannot generate render_id; missing fields: " + ", ".join(sorted(missing))
        )

    payload = {
        "version": version,
        "fields": {field: _canonical_value(record[field]) for field in fields},
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[: int(digest_size)]
    return f"{prefix}_{digest}" if prefix else digest

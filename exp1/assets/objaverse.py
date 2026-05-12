"""Objaverse asset access helpers with explicit storage limits."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from exp1.assets.shapenet import infer_photorealistic_material_available


LVIS_CATEGORY_ALIASES = {
    "automobile": "car_(automobile)",
    "car": "car_(automobile)",
}


def bytes_from_gb(value: Optional[Union[str, float, int]]) -> Optional[int]:
    """Convert a GiB limit to bytes."""
    if value is None or str(value).strip() == "":
        return None
    return int(float(value) * 1024**3)


def _object_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return 0


def _category_values(categories: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if categories is None:
        return None
    if isinstance(categories, str):
        raw = [part.strip() for part in categories.split(",")]
    else:
        raw = [str(part).strip() for part in categories]
    values = [value for value in raw if value]
    return values or None


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_lvis_categories(
    annotations: Mapping[str, Sequence[str]],
    categories: Optional[Union[str, Sequence[str]]] = None,
) -> List[str]:
    """Resolve user-facing category names to exact Objaverse LVIS keys."""
    requested = _category_values(categories)
    if requested is None:
        return sorted(str(category) for category in annotations)

    by_lower = {str(category).lower(): str(category) for category in annotations}
    selected_categories = []
    for category in requested:
        key = category.lower()
        key = LVIS_CATEGORY_ALIASES.get(key, key)
        if key not in by_lower:
            raise KeyError(f"Objaverse LVIS category not found: {category}")
        selected_categories.append(by_lower[key])
    return selected_categories


def select_objaverse_lvis_uids(
    annotations: Mapping[str, Sequence[str]],
    *,
    categories: Optional[Union[str, Sequence[str]]] = None,
    max_objects: Optional[int] = None,
) -> List[str]:
    """Select deterministic Objaverse UIDs from LVIS category annotations."""
    selected_categories = _resolve_lvis_categories(annotations, categories)

    uids = _dedupe(
        uid
        for category in selected_categories
        for uid in annotations.get(category, ())
    )
    if max_objects is not None:
        uids = uids[: int(max_objects)]
    return uids


def _category_for_uid(
    uid: str,
    annotations: Mapping[str, Sequence[str]],
    requested_categories: Optional[Sequence[str]],
) -> str:
    categories = (
        list(requested_categories)
        if requested_categories is not None
        else sorted(str(category) for category in annotations)
    )
    for category in categories:
        if uid in set(annotations.get(category, ())):
            return str(category)
    return "unknown"


def _asset_row(
    *,
    uid: str,
    path: Path,
    category: str,
    split: str,
    cache_bytes: int,
) -> Dict[str, Any]:
    return {
        "object_id": f"objaverse_{uid}",
        "source_dataset": "objaverse",
        "category": category,
        "split": split,
        "raw_mesh_path": str(path),
        "normalized_mesh_path": "",
        "asset_status": "discovered",
        "asset_error_message": "",
        "objaverse_uid": uid,
        "objaverse_cache_bytes": int(cache_bytes),
        "has_photorealistic_material": infer_photorealistic_material_available(path),
    }


def download_objaverse_assets(
    *,
    cache_dir: Union[str, Path],
    categories: Optional[Union[str, Sequence[str]]] = None,
    max_objects: int,
    max_download_bytes: Optional[int],
    download_processes: int = 4,
    batch_size: int = 8,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Download a bounded Objaverse LVIS subset and return asset manifest rows.

    The Objaverse API does not expose file sizes before download, so the byte
    limit is enforced incrementally after each small batch. Keep ``batch_size``
    modest to avoid overshooting the local cache limit by much.
    """
    try:
        import objaverse
    except ImportError as exc:
        raise ImportError(
            "Objaverse downloads require the `objaverse` package. "
            "Install requirements.txt or run `pip install objaverse`."
        ) from exc

    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OBJAVERSE_HOME", str(cache_dir))

    annotations = objaverse.load_lvis_annotations()
    requested_categories = _resolve_lvis_categories(annotations, categories)
    uids = select_objaverse_lvis_uids(
        annotations,
        categories=requested_categories,
        max_objects=max_objects,
    )

    rows: List[Dict[str, Any]] = []
    total_bytes = 0
    for start in range(0, len(uids), max(1, int(batch_size))):
        batch = uids[start : start + max(1, int(batch_size))]
        objects = objaverse.load_objects(
            uids=batch,
            download_processes=int(download_processes),
        )
        for uid in batch:
            path_value = objects.get(uid)
            if path_value is None:
                continue
            path = Path(path_value).expanduser().resolve()
            size = _object_size(path)
            if (
                max_download_bytes is not None
                and total_bytes + size > int(max_download_bytes)
            ):
                return rows
            total_bytes += size
            rows.append(
                _asset_row(
                    uid=uid,
                    path=path,
                    category=_category_for_uid(
                        uid,
                        annotations,
                        requested_categories,
                    ),
                    split=split,
                    cache_bytes=size,
                )
            )
            if len(rows) >= int(max_objects):
                return rows
    return rows

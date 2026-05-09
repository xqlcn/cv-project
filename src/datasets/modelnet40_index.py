"""Pure filesystem index helpers for ModelNet40 meshes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


__all__ = [
    "default_modelnet40_root",
    "discover_modelnet40_records",
    "load_records_json",
    "save_records_json",
]


def default_modelnet40_root(project_root: Optional[Union[str, Path]] = None) -> Path:
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    return root / "data" / "modelnet40"


def discover_modelnet40_records(
    root: Union[str, Path],
    *,
    splits: Sequence[str] = ("train", "test"),
) -> List[Dict[str, Any]]:
    """
    Scan ``root/{split}/{category}/*.off`` (normalized layout).

    If only the raw Princeton tree exists (``root/ModelNet40/<category>/{split}/``),
    records are still discovered so you can list data before normalizing.

    Each record:
        mesh_path, category, split, object_id, dataset="modelnet40"
    """
    root = Path(root).expanduser().resolve()
    records: List[Dict[str, Any]] = []

    def add_from_split_category(split: str, category: str, cat_dir: Path) -> None:
        for off in sorted(cat_dir.glob("*.off")):
            stem = off.stem
            oid = f"{category}_{stem}"
            records.append(
                {
                    "mesh_path": str(off.resolve()),
                    "category": category,
                    "split": split,
                    "object_id": oid,
                    "dataset": "modelnet40",
                }
            )

    for split in splits:
        split_dir = root / split
        if split_dir.is_dir():
            for cat_dir in sorted(split_dir.iterdir()):
                if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                    continue
                add_from_split_category(split, cat_dir.name, cat_dir)

    if records:
        records.sort(key=lambda r: (r["split"], r["category"], r["object_id"]))
        return records

    bundled = root / "ModelNet40"
    if bundled.is_dir():
        for cat_dir in sorted(bundled.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                continue
            category = cat_dir.name
            for split in splits:
                split_dir = cat_dir / split
                if split_dir.is_dir():
                    add_from_split_category(split, category, split_dir)

    records.sort(key=lambda r: (r["split"], r["category"], r["object_id"]))
    return records


def save_records_json(
    path: Union[str, Path],
    records: Sequence[Dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=2) + "\n", encoding="utf-8")


def load_records_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        return list(data["records"])
    return list(data)

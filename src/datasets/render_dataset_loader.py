"""Load rendered dataset metadata (JSONL) with optional filters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.io import read_jsonl


@dataclass
class RenderRecord:
    raw: Dict[str, Any]

    @property
    def render_path(self) -> Path:
        return Path(self.raw["render_path"])

    @property
    def object_id(self) -> str:
        return str(self.raw["object_id"])

    @property
    def chirality_label(self) -> int:
        return int(self.raw["chirality_label"])

    @property
    def sample_kind(self) -> str:
        return str(self.raw.get("sample_kind", "unknown"))


def load_chirality_records(
    index_path: Path,
    *,
    include_kinds: Optional[List[str]] = None,
) -> List[RenderRecord]:
    rows = read_jsonl(index_path)
    records = [RenderRecord(r) for r in rows]
    if include_kinds is not None:
        records = [r for r in records if r.sample_kind in include_kinds]
    return records

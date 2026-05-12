"""Image dataset wrappers for Experiment 1 render manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import pandas as pd
from PIL import Image

from exp1.data.splits import filter_manifest
from exp1.metadata.manifest import load_manifest


RowsLike = Union[pd.DataFrame, Iterable[Mapping[str, Any]]]


def _as_dataframe(rows_or_path: Union[RowsLike, str, Path]) -> pd.DataFrame:
    if isinstance(rows_or_path, (str, Path)):
        return load_manifest(rows_or_path, validate=False)
    if isinstance(rows_or_path, pd.DataFrame):
        return rows_or_path.copy()
    return pd.DataFrame(list(rows_or_path))


def _resolve_path(project_root: Optional[Union[str, Path]], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


class Exp1ImageDataset:
    """Load RGB render images with manifest metadata and simple filters."""

    def __init__(
        self,
        manifest_rows: Union[RowsLike, str, Path],
        *,
        split: Optional[Union[str, list[str]]] = None,
        texture_condition: Optional[Union[str, list[str]]] = None,
        require_qc_pass: bool = True,
        image_column: str = "rgb_path",
        project_root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[Image.Image], Any]] = None,
    ) -> None:
        df = _as_dataframe(manifest_rows)
        self.df = filter_manifest(
            df,
            split=split,
            texture_condition=texture_condition,
            require_qc_pass=require_qc_pass,
        )
        self.image_column = image_column
        self.project_root = project_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[int(index)].to_dict()
        image_path = _resolve_path(self.project_root, row[self.image_column])
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            image_value = self.transform(rgb) if self.transform else rgb.copy()
        return {
            "image": image_value,
            "render_id": row["render_id"],
            "object_id": row.get("object_id"),
            "split": row.get("split"),
            "texture_condition": row.get("texture_condition"),
            "metadata": row,
        }

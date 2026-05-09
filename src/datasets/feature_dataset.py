"""Dataset that loads precomputed .pt feature files for probing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class FeatureProbeDataset(Dataset):
    """
    Each item returns (feature_tensor, label_tensor, meta_dict).

    feature_tensor is taken from the saved dict key:
      - cls_final, or
      - patch_mean, or
      - layer_cls[layer]
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        *,
        layer: Optional[int],
        token_key: str = "cls_final",
    ) -> None:
        self.rows = rows
        self.layer = layer
        self.token_key = token_key

    def __len__(self) -> int:
        return len(self.rows)

    def _load_feat(self, path: Path) -> torch.Tensor:
        blob = torch.load(path, map_location="cpu")
        if self.layer is not None:
            layer_map = blob.get("layer_cls")
            if layer_map is None or self.layer not in layer_map:
                raise KeyError(f"Missing layer {self.layer} in {path}")
            return layer_map[self.layer].float()
        if self.token_key not in blob:
            raise KeyError(f"Missing {self.token_key} in {path}")
        return blob[self.token_key].float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.rows[idx]
        feat_path = Path(row["feature_path"])
        x = self._load_feat(feat_path)
        y = torch.tensor(row["label"], dtype=torch.long)
        return x, y, row

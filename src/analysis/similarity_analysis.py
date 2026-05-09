"""
Cosine similarity diagnostics for chirality pairs vs viewpoint jitter.

Reads `features_index.jsonl` produced by `extract_features.py`, groups rows by
`pair_group_id`, and compares:
  - original vs mirrored chirality
  - original vs small viewpoint shift (same mesh)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.metrics import cosine_similarity_pairs
from src.utils.seed import set_seed


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_vector(blob: Dict[str, Any], *, layer: int | None, token_key: str) -> torch.Tensor:
    if layer is not None:
        m = blob.get("layer_cls")
        if m is None or layer not in m:
            raise KeyError(f"layer_cls missing layer {layer}")
        return m[layer].float().view(1, -1)
    if token_key not in blob:
        raise KeyError(f"missing {token_key}")
    return blob[token_key].float().view(1, -1)


@hydra.main(version_base=None, config_path="../../configs", config_name="clip_probe")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    project_root = Path(cfg.paths.project_root).resolve()
    set_seed(int(cfg.probe.seed))

    feature_root = Path(cfg.paths.feature_dir)
    if not feature_root.is_absolute():
        feature_root = project_root / feature_root
    index_path = feature_root / "features_index.jsonl"
    rows = _load_rows(index_path)

    grouped: DefaultDict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        pid = str(r.get("pair_group_id", ""))
        kind = str(r.get("sample_kind", ""))
        if not pid or kind not in {"chirality_original", "chirality_mirror", "view_jitter"}:
            continue
        grouped[pid][kind] = r

    token = str(cfg.features.token)
    token_key = "cls_final" if token == "cls" else "patch_mean"
    tl = OmegaConf.select(cfg, "probe.target_layer")
    layer = int(tl) if tl is not None else None

    sims_mirror: List[float] = []
    sims_view: List[float] = []
    skipped = 0

    for pid, items in grouped.items():
        if "chirality_original" not in items:
            skipped += 1
            continue
        orig_path = Path(items["chirality_original"]["feature_path"])
        z0 = _load_vector(torch.load(orig_path, map_location="cpu"), layer=layer, token_key=token_key)

        if "chirality_mirror" in items:
            mp = Path(items["chirality_mirror"]["feature_path"])
            zm = _load_vector(torch.load(mp, map_location="cpu"), layer=layer, token_key=token_key)
            sims_mirror.append(float(cosine_similarity_pairs(z0, zm).item()))

        if "view_jitter" in items:
            jp = Path(items["view_jitter"]["feature_path"])
            zj = _load_vector(torch.load(jp, map_location="cpu"), layer=layer, token_key=token_key)
            sims_view.append(float(cosine_similarity_pairs(z0, zj).item()))

    def _summ(name: str, xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"count": 0}
        t = torch.tensor(xs)
        return {
            "count": float(len(xs)),
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    report = {
        "original_vs_mirror": _summ("mirror", sims_mirror),
        "original_vs_view_jitter": _summ("view", sims_view),
        "skipped_groups_missing_original": skipped,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

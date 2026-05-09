"""Extract frozen backbone features for every render in a metadata JSONL index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models.clip_extractor import CLIPExtractorConfig, FrozenCLIPExtractor
from src.models.dino_extractor import DINOExtractorConfig, FrozenDINOv2Extractor
from src.utils.io import ensure_dir, write_jsonl
from src.utils.seed import set_seed


def _rows_from_metadata(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@hydra.main(version_base=None, config_path="../../configs", config_name="clip_probe")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    project_root = Path(cfg.paths.project_root).resolve()
    set_seed(int(cfg.probe.seed))

    meta_path = Path(cfg.paths.metadata_index)
    if not meta_path.is_file():
        meta_path = project_root / meta_path
    rows = _rows_from_metadata(meta_path)
    limit = cfg.train.limit_samples if "train" in cfg and cfg.train.limit_samples is not None else None
    if limit is not None:
        rows = rows[: int(limit)]

    feature_root = Path(cfg.paths.feature_dir)
    if not feature_root.is_absolute():
        feature_root = project_root / feature_root
    ensure_dir(feature_root)

    device = torch.device(cfg.features.device if torch.cuda.is_available() else "cpu")
    token = str(cfg.features.token)

    image_paths = [str(Path(row["render_path"]).resolve()) for r in rows]

    if cfg.backbone.family == "clip":
        ext = FrozenCLIPExtractor(
            CLIPExtractorConfig(
                model_id=str(cfg.backbone.model_id),
                image_size=int(cfg.backbone.image_size),
                use_open_clip=bool(cfg.backbone.use_open_clip),
                open_clip_model=str(cfg.backbone.open_clip_model),
                open_clip_pretrained=str(cfg.backbone.open_clip_pretrained),
                layers=list(cfg.features.layers) if cfg.features.layers is not None else None,
            ),
            device=device,
        )
    elif cfg.backbone.family == "dino":
        ext = FrozenDINOv2Extractor(
            DINOExtractorConfig(
                hub_name=str(cfg.backbone.hub_name),
                image_size=int(cfg.backbone.image_size),
                layers=list(cfg.features.layers) if cfg.features.layers is not None else None,
            ),
            device=device,
        )
    else:
        raise ValueError(f"Unknown backbone.family: {cfg.backbone.family}")

    feats = ext.extract_paths(
        image_paths,
        batch_size=int(cfg.features.batch_size),
        token=token,  # type: ignore[arg-type]
    )

    index_out: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="save features"):
        rp = str(Path(row["render_path"]).resolve())
        if rp not in feats:
            raise KeyError(f"No features for render_path={rp}")
        payload = feats[rp]
        out_name = f"{Path(row['render_path']).stem}.pt"
        out_path = feature_root / out_name
        torch.save(payload, out_path)
        merged = dict(row)
        merged["feature_path"] = str(out_path.resolve())
        index_out.append(merged)

    index_path = feature_root / "features_index.jsonl"
    write_jsonl(index_path, index_out)
    print(f"Saved {len(index_out)} feature files to {feature_root}")
    print(f"Wrote index {index_path}")


if __name__ == "__main__":
    main()

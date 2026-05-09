"""Evaluate a saved linear probe checkpoint on the validation split (recomputed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.feature_dataset import FeatureProbeDataset
from src.models.linear_probe import LinearProbe, ProbeConfig
from src.utils.metrics import binary_classification_metrics
from src.utils.seed import set_seed


def _load_feature_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _filter_chirality(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("sample_kind") not in {"chirality_original", "chirality_mirror"}:
            continue
        label = int(r["chirality_label"])
        if label < 0:
            continue
        row = dict(r)
        row["label"] = label
        out.append(row)
    return out


def collate(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


@hydra.main(version_base=None, config_path="../../configs", config_name="clip_probe")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    project_root = Path(cfg.paths.project_root).resolve()
    set_seed(int(cfg.probe.seed))

    out_dir = Path(cfg.paths.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    ckpt_path = out_dir / "linear_probe.pt"
    ckpt_override = OmegaConf.select(cfg, "paths.checkpoint")
    if ckpt_override is not None:
        ckpt_path = Path(ckpt_override)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location="cpu")
    probe_cfg = ProbeConfig(**blob["probe_cfg"])
    model = LinearProbe(probe_cfg)
    model.load_state_dict(blob["state_dict"])
    device = torch.device(cfg.features.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    feature_root = Path(cfg.paths.feature_dir)
    if not feature_root.is_absolute():
        feature_root = project_root / feature_root
    index_path = feature_root / "features_index.jsonl"
    rows = _filter_chirality(_load_feature_index(index_path))
    _, val_rows = train_test_split(
        rows,
        test_size=float(cfg.probe.val_fraction),
        random_state=int(cfg.probe.seed),
        stratify=[r["label"] for r in rows],
    )

    token = str(cfg.features.token)
    token_key = "cls_final" if token == "cls" else "patch_mean"
    tl = OmegaConf.select(cfg, "probe.target_layer")
    layer = int(tl) if tl is not None else None

    val_ds = FeatureProbeDataset(val_rows, layer=layer, token_key=token_key)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.features.batch_size),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(cfg.features.num_workers),
    )

    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for xb, yb in tqdm(val_loader, desc="eval"):
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_labels.append(yb)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = binary_classification_metrics(logits_cat, labels_cat)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

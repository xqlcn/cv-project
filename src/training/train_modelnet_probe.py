"""Train a multiclass linear probe on ModelNet40 rendered-view features."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.feature_dataset import FeatureProbeDataset
from src.models.linear_probe import LinearProbe, ProbeConfig
from src.utils.io import ensure_dir
from src.utils.seed import seed_worker, set_seed


def collate_xy(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


def _worker_init_fn(worker_id: int, base_seed: int) -> None:
    seed_worker(worker_id, base_seed)


def _load_feature_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _filter_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if "category_id" not in r:
            continue
        row = dict(r)
        row["label"] = int(r["category_id"])
        out.append(row)
    return out


def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


@hydra.main(version_base=None, config_path="../../configs", config_name="modelnet_clip")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    project_root = Path(cfg.paths.project_root).resolve()
    set_seed(int(cfg.probe.seed))

    feature_root = Path(cfg.paths.feature_dir)
    if not feature_root.is_absolute():
        feature_root = project_root / feature_root
    index_path = feature_root / "features_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing feature index: {index_path}. Run extract_features first.")

    rows = _filter_rows(_load_feature_index(index_path))
    if not rows:
        raise RuntimeError("No rows with category_id found in features index.")

    limit = OmegaConf.select(cfg, "train.limit_samples")
    if limit is not None:
        rows = rows[: int(limit)]

    train_rows, val_rows = train_test_split(
        rows,
        test_size=float(cfg.probe.val_fraction),
        random_state=int(cfg.probe.seed),
        stratify=[r["label"] for r in rows],
    )

    token = str(cfg.features.token)
    token_key = "cls_final" if token == "cls" else "patch_mean"
    tl = OmegaConf.select(cfg, "probe.target_layer")
    layer: Optional[int] = int(tl) if tl is not None else None

    train_ds = FeatureProbeDataset(train_rows, layer=layer, token_key=token_key)
    val_ds = FeatureProbeDataset(val_rows, layer=layer, token_key=token_key)
    x0, _, _ = train_ds[0]

    n_classes = int(cfg.probe.num_classes)
    probe_cfg = ProbeConfig(
        input_dim=int(x0.numel()),
        num_outputs=n_classes,
        task="multiclass",
        use_layernorm=bool(cfg.probe.use_layernorm),
        logistic=False,
    )
    model = LinearProbe(probe_cfg)
    device = torch.device(cfg.features.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.probe.lr),
        weight_decay=float(cfg.probe.weight_decay),
    )

    g = torch.Generator()
    g.manual_seed(int(cfg.probe.seed))
    worker_init = partial(_worker_init_fn, base_seed=int(cfg.probe.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.features.batch_size),
        shuffle=True,
        collate_fn=collate_xy,
        num_workers=int(cfg.features.num_workers),
        worker_init_fn=worker_init,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.features.batch_size),
        shuffle=False,
        collate_fn=collate_xy,
        num_workers=int(cfg.features.num_workers),
    )

    out_dir = Path(cfg.paths.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    ensure_dir(out_dir)

    best = -1.0
    best_state = None
    for epoch in range(int(cfg.probe.epochs)):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"train {epoch}"):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        all_logits = []
        all_labels = []
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.append(logits.cpu())
                all_labels.append(yb)
        lcat = torch.cat(all_logits, dim=0)
        ycat = torch.cat(all_labels, dim=0)
        acc = _acc(lcat, ycat)
        print(f"epoch={epoch} train_loss={sum(losses)/max(1,len(losses)):.4f} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        ckpt_path = out_dir / "linear_probe_modelnet40.pt"
        torch.save(
            {
                "state_dict": best_state,
                "probe_cfg": asdict(probe_cfg),
                "best_val_acc": best,
                "train_config": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

"""Train a linear azimuth probe on frozen features (circular regression with sin/cos targets)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.linear_probe import LinearProbe, ProbeConfig
from src.utils.io import ensure_dir
from src.utils.seed import seed_worker, set_seed


class AzimuthFeatureDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], *, layer: Optional[int], token_key: str) -> None:
        self.rows = rows
        self.layer = layer
        self.token_key = token_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        feat = torch.load(Path(row["feature_path"]), map_location="cpu")
        if self.layer is not None:
            x = feat["layer_cls"][self.layer].float()
        else:
            x = feat[self.token_key].float()
        az = float(row["azimuth"])
        y = torch.tensor([math.sin(az), math.cos(az)], dtype=torch.float32)
        return x, y, row


def _load_feature_index(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collate_xy(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


def _worker_init_fn(worker_id: int, base_seed: int) -> None:
    seed_worker(worker_id, base_seed)


def _angular_mae_deg(pred_sc: torch.Tensor, targ_sc: torch.Tensor) -> float:
    pred_ang = torch.atan2(pred_sc[:, 0], pred_sc[:, 1])
    targ_ang = torch.atan2(targ_sc[:, 0], targ_sc[:, 1])
    diff = torch.atan2(torch.sin(pred_ang - targ_ang), torch.cos(pred_ang - targ_ang))
    return float(torch.rad2deg(torch.abs(diff)).mean().item())


@hydra.main(version_base=None, config_path="../../configs", config_name="modelnet_viewpoint_clip")
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

    rows = [r for r in _load_feature_index(index_path) if "azimuth" in r]
    if not rows:
        raise RuntimeError("No azimuth field found in feature rows.")

    limit = OmegaConf.select(cfg, "train.limit_samples")
    if limit is not None:
        rows = rows[: int(limit)]

    train_rows, val_rows = train_test_split(
        rows,
        test_size=float(cfg.probe.val_fraction),
        random_state=int(cfg.probe.seed),
        shuffle=True,
    )

    token = str(cfg.features.token)
    token_key = "cls_final" if token == "cls" else "patch_mean"
    tl = OmegaConf.select(cfg, "probe.target_layer")
    layer: Optional[int] = int(tl) if tl is not None else None

    train_ds = AzimuthFeatureDataset(train_rows, layer=layer, token_key=token_key)
    val_ds = AzimuthFeatureDataset(val_rows, layer=layer, token_key=token_key)
    x0, _, _ = train_ds[0]

    probe_cfg = ProbeConfig(
        input_dim=int(x0.numel()),
        num_outputs=2,  # sin, cos
        task="regression",
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

    best = float("inf")
    best_state = None
    for epoch in range(int(cfg.probe.epochs)):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"train {epoch}"):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        pred_all = []
        targ_all = []
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred_all.append(model(xb).cpu())
                targ_all.append(yb)
        pcat = torch.cat(pred_all, dim=0)
        tcat = torch.cat(targ_all, dim=0)
        mae_deg = _angular_mae_deg(pcat, tcat)
        print(f"epoch={epoch} train_loss={sum(losses)/max(1,len(losses)):.5f} val_azimuth_mae_deg={mae_deg:.3f}")

        if mae_deg < best:
            best = mae_deg
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        ckpt = out_dir / "linear_probe_azimuth.pt"
        torch.save(
            {
                "state_dict": best_state,
                "probe_cfg": asdict(probe_cfg),
                "best_val_azimuth_mae_deg": best,
                "train_config": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt,
        )
        print(f"Saved checkpoint to {ckpt}")


if __name__ == "__main__":
    main()

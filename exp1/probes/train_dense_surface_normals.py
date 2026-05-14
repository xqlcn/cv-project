"""Train dense per-patch surface-normal probes on cached patch features."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from exp1.data.splits import assert_object_disjoint_splits
from exp1.metadata.manifest import load_manifest
from exp1.probes.dense_surface_normals import (
    DenseSurfaceNormalHead,
    DenseSurfaceNormalHeadConfig,
    dense_surface_normal_loss,
    dense_surface_normal_metrics,
    dense_surface_normal_sample_metrics,
)
from exp1.tasks.dense_surface_normals import Exp1DenseSurfaceNormalDataset

__all__ = [
    "DenseSurfaceNormalTrainConfig",
    "evaluate_dense_surface_normal_arrays",
    "train_dense_surface_normal_probe",
]


@dataclass
class DenseSurfaceNormalTrainConfig:
    """Hyperparameters for dense surface-normal linear probe training."""

    task: str = "dense_surface_normal_patches"
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"
    use_layernorm: bool = True
    scheduler: str = "cosine_warmup"
    warmup_epochs: float = 4.0
    monitor: str = "val_angular_error_deg_mean"
    feature_mode: str = "patch"
    target_mode: str = "camera_frame_normals"
    min_valid_fraction_per_patch: float = 0.25
    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 0.0


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    requested = str(device)
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        requested = "cpu"
    return torch.device(requested)


def _make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    m = torch.from_numpy(np.asarray(valid, dtype=bool))
    dataset = TensorDataset(x, y, m)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
    )


def evaluate_dense_surface_normal_arrays(
    model: torch.nn.Module,
    *,
    features: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    device: Union[str, torch.device],
    batch_size: int = 32,
    max_samples_for_metrics: Optional[int] = None,
    return_per_sample: bool = False,
) -> Union[
    tuple[np.ndarray, Mapping[str, float]],
    tuple[np.ndarray, Mapping[str, float], list[dict[str, float]]],
]:
    """Run dense-normal inference and compute angular-error metrics."""
    device = _resolve_device(device)
    loader = _make_loader(
        features, targets, valid, batch_size=batch_size, shuffle=False, seed=0
    )
    model.eval()
    preds: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for xb, yb, mb in loader:
            y = model(xb.to(device))
            preds.append(y.detach().cpu())
            target_chunks.append(yb)
            mask_chunks.append(mb)
    output_tensor = torch.cat(preds, dim=0)
    target_tensor = torch.cat(target_chunks, dim=0)
    mask_tensor = torch.cat(mask_chunks, dim=0)
    if (
        max_samples_for_metrics is not None
        and output_tensor.shape[0] > int(max_samples_for_metrics)
    ):
        n_keep = int(max_samples_for_metrics)
        indices = np.linspace(0, output_tensor.shape[0] - 1, n_keep).astype(int)
        out_for_metrics = output_tensor[indices]
        tgt_for_metrics = target_tensor[indices]
        msk_for_metrics = mask_tensor[indices]
    else:
        out_for_metrics = output_tensor
        tgt_for_metrics = target_tensor
        msk_for_metrics = mask_tensor
    metrics = dense_surface_normal_metrics(
        out_for_metrics,
        tgt_for_metrics,
        msk_for_metrics,
    )
    if return_per_sample:
        sample_metrics = dense_surface_normal_sample_metrics(
            output_tensor,
            target_tensor,
            mask_tensor,
        )
        return output_tensor.numpy(), dict(metrics), sample_metrics
    return output_tensor.numpy(), dict(metrics)


def _cosine_warmup_lambda(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress_den = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(progress_den)))
    return 0.5 * (1.0 + float(np.cos(np.pi * progress)))


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    cfg: DenseSurfaceNormalTrainConfig,
    steps_per_epoch: int,
) -> Optional[LambdaLR]:
    if str(cfg.scheduler).lower() in {"", "none", "constant"}:
        return None
    if str(cfg.scheduler).lower() != "cosine_warmup":
        raise ValueError(f"Unsupported dense-normal scheduler: {cfg.scheduler}")
    total_steps = max(1, int(cfg.epochs) * max(1, int(steps_per_epoch)))
    warmup_steps = int(round(float(cfg.warmup_epochs) * max(1, int(steps_per_epoch))))
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_warmup_lambda(
            int(step),
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        ),
    )


def _monitor_value(
    row: Mapping[str, float],
    *,
    cfg: DenseSurfaceNormalTrainConfig,
) -> float:
    candidates = [str(cfg.monitor)]
    if not str(cfg.monitor).startswith(("train_", "val_", "test_")):
        candidates = [f"val_{cfg.monitor}", f"train_{cfg.monitor}", str(cfg.monitor)]
    for key in candidates:
        if key in row:
            return float(row[key])
    return float("nan")


def _split_arrays(
    patch_cache: Union[str, Path],
    manifest: pd.DataFrame,
    *,
    split: str,
    texture_condition: Optional[Sequence[str]],
    project_root: Optional[Union[str, Path]],
    cfg: DenseSurfaceNormalTrainConfig,
) -> Optional[dict[str, Any]]:
    dataset = Exp1DenseSurfaceNormalDataset(
        patch_cache,
        manifest=manifest,
        split=split,
        texture_condition=texture_condition,
        project_root=project_root,
        min_valid_fraction_per_patch=float(cfg.min_valid_fraction_per_patch),
        feature_mode=str(cfg.feature_mode),
    )
    arrays = dataset.materialize_arrays()
    if len(arrays["render_ids"]) == 0:
        return None
    arrays["patch_grid_shape"] = dataset.patch_grid_shape
    arrays["feature_dim"] = dataset.feature_dim
    return arrays


def _save_predictions(
    output_dir: Path,
    split_name: str,
    arrays: Mapping[str, Any],
    predictions: np.ndarray,
) -> Path:
    out = output_dir / f"predictions_{split_name}.npz"
    np.savez_compressed(
        out,
        render_ids=np.asarray(arrays["render_ids"], dtype=str),
        targets=np.asarray(arrays["targets"], dtype=np.float32),
        valid=np.asarray(arrays["valid"], dtype=bool),
        predictions=np.asarray(predictions, dtype=np.float32),
    )
    return out


def _assert_texture_control_groups_single_split(manifest: pd.DataFrame) -> None:
    if "texture_control_group_id" not in manifest.columns:
        return
    required = {"texture_control_group_id", "split"}
    if not required.issubset(manifest.columns):
        return
    split_counts = manifest.groupby("texture_control_group_id")["split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        preview = ", ".join(str(value) for value in leaked.index[:8])
        raise ValueError(
            "Texture-control groups appear in multiple splits: " + preview
        )


def _prediction_summary_frame(
    *,
    split_name: str,
    arrays: Mapping[str, Any],
    sample_metrics: Sequence[Mapping[str, float]],
) -> pd.DataFrame:
    rows = arrays["rows"].reset_index(drop=True).copy()
    keep_columns = [
        column
        for column in (
            "render_id",
            "object_id",
            "source_dataset",
            "category",
            "texture_condition",
            "texture_control_group_id",
            "split",
        )
        if column in rows.columns
    ]
    out = rows.loc[:, keep_columns].copy()
    if "split" not in out.columns:
        out["split"] = split_name
    metric_df = pd.DataFrame(list(sample_metrics))
    if len(metric_df) != len(out):
        raise ValueError(
            f"Got {len(metric_df)} per-sample metrics for {len(out)} prediction rows"
        )
    return pd.concat([out.reset_index(drop=True), metric_df], axis=1)


def train_dense_surface_normal_probe(
    *,
    patch_cache: Union[str, Path],
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str,
    layer_name: str,
    texture_condition: Optional[Union[str, Sequence[str]]] = None,
    train_texture_condition: Optional[Union[str, Sequence[str]]] = None,
    eval_texture_condition: Optional[Union[str, Sequence[str]]] = None,
    project_root: Optional[Union[str, Path]] = None,
    config: Optional[DenseSurfaceNormalTrainConfig] = None,
) -> dict[str, Any]:
    """Train and evaluate one dense surface-normal probe."""
    cfg = copy.deepcopy(config) if config is not None else DenseSurfaceNormalTrainConfig()
    _seed_everything(int(cfg.seed))
    device = _resolve_device(cfg.device)

    manifest = load_manifest(manifest_path, validate=False)
    assert_object_disjoint_splits(manifest)
    _assert_texture_control_groups_single_split(manifest)

    train_texture = train_texture_condition or texture_condition
    eval_texture = eval_texture_condition or texture_condition
    split_arrays: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        split_texture = (
            train_texture if split_name in {"train", "val"} else eval_texture
        )
        if isinstance(split_texture, str):
            split_texture_list: Optional[Sequence[str]] = [split_texture]
        else:
            split_texture_list = list(split_texture) if split_texture else None
        arrays = _split_arrays(
            patch_cache,
            manifest,
            split=split_name,
            texture_condition=split_texture_list,
            project_root=project_root,
            cfg=cfg,
        )
        if arrays is not None:
            split_arrays[split_name] = arrays

    if "train" not in split_arrays:
        raise ValueError(
            "No dense-normal train rows are available after feature/manifest joins"
        )
    train_arrays = split_arrays["train"]
    val_arrays = split_arrays.get("val")

    head_cfg = DenseSurfaceNormalHeadConfig(
        feature_dim=int(train_arrays["feature_dim"]),
        use_layernorm=bool(cfg.use_layernorm),
    )
    model = DenseSurfaceNormalHead(head_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    train_loader = _make_loader(
        train_arrays["features"],
        train_arrays["targets"],
        train_arrays["valid"],
        batch_size=int(cfg.batch_size),
        shuffle=True,
        seed=int(cfg.seed),
    )
    scheduler = _make_scheduler(
        optimizer,
        cfg=cfg,
        steps_per_epoch=len(train_loader),
    )

    history: list[dict[str, float]] = []
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_score: Optional[float] = None
    bad_epochs = 0

    def _better(value: float, current_best: Optional[float]) -> bool:
        if current_best is None:
            return True
        return value < current_best - float(cfg.min_delta)

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        losses: list[float] = []
        for xb, yb, mb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb.to(device))
            loss = dense_surface_normal_loss(pred, yb.to(device), mb.to(device))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach().cpu().item()))

        _, train_metrics = evaluate_dense_surface_normal_arrays(
            model,
            features=train_arrays["features"],
            targets=train_arrays["targets"],
            valid=train_arrays["valid"],
            device=device,
            batch_size=int(cfg.batch_size),
            max_samples_for_metrics=256,
        )
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})

        if val_arrays is not None:
            _, val_metrics = evaluate_dense_surface_normal_arrays(
                model,
                features=val_arrays["features"],
                targets=val_arrays["targets"],
                valid=val_arrays["valid"],
                device=device,
                batch_size=int(cfg.batch_size),
                max_samples_for_metrics=256,
            )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        monitor_metric = _monitor_value(row, cfg=cfg)
        if np.isfinite(monitor_metric) and _better(monitor_metric, best_score):
            best_score = monitor_metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        if cfg.early_stopping and val_arrays is not None and bad_epochs >= int(cfg.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Mapping[str, float]] = {}
    prediction_paths: dict[str, Path] = {}
    prediction_summary_frames: list[pd.DataFrame] = []
    for split_name, arrays in split_arrays.items():
        predictions, split_metrics, sample_metrics = evaluate_dense_surface_normal_arrays(
            model,
            features=arrays["features"],
            targets=arrays["targets"],
            valid=arrays["valid"],
            device=device,
            batch_size=int(cfg.batch_size),
            return_per_sample=True,
        )
        metrics[split_name] = split_metrics
        prediction_paths[split_name] = _save_predictions(
            output_dir,
            split_name,
            arrays,
            predictions,
        )
        prediction_summary_frames.append(
            _prediction_summary_frame(
                split_name=split_name,
                arrays=arrays,
                sample_metrics=sample_metrics,
            )
        )

    predictions_csv_path = output_dir / "predictions.csv"
    pd.concat(prediction_summary_frames, ignore_index=True).to_csv(
        predictions_csv_path,
        index=False,
    )
    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "head_config": asdict(head_cfg),
            "train_config": asdict(cfg),
            "metrics": metrics,
            "model_name": model_name,
            "layer_name": layer_name,
        },
        checkpoint_path,
    )
    metadata = {
        "task": cfg.task,
        "model_name": model_name,
        "layer_name": layer_name,
        "patch_cache": str(patch_cache),
        "manifest_path": str(manifest_path),
        "texture_condition": texture_condition,
        "train_texture_condition": train_texture_condition,
        "eval_texture_condition": eval_texture_condition,
        "feature_mode": cfg.feature_mode,
        "target_mode": cfg.target_mode,
        "loss": "masked_cosine",
        "scheduler": cfg.scheduler,
        "warmup_epochs": float(cfg.warmup_epochs),
        "monitor": cfg.monitor,
        "min_valid_fraction_per_patch": float(cfg.min_valid_fraction_per_patch),
        "patch_grid_shape": list(train_arrays["patch_grid_shape"]),
        "feature_dim": int(train_arrays["feature_dim"]),
        "num_rows_by_split": {
            split_name: int(len(arrays["render_ids"]))
            for split_name, arrays in split_arrays.items()
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"metrics": metrics, "metadata": metadata},
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "metrics": metrics,
        "history": history,
        "metadata": metadata,
        "artifact_paths": {
            "checkpoint": checkpoint_path,
            "metrics": metrics_path,
            "history": history_path,
            "predictions": predictions_csv_path,
            **{f"predictions_{name}": path for name, path in prediction_paths.items()},
        },
        "train_config": asdict(cfg),
    }

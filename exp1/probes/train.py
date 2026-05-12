"""Train linear probes on cached Experiment 1 features."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from exp1.data.feature_dataset import Exp1FeatureDataset
from exp1.data.splits import assert_object_disjoint_splits
from exp1.metadata.manifest import load_manifest
from exp1.probes.checkpoints import (
    save_history_csv,
    save_metrics_json,
    save_predictions_csv,
    save_probe_checkpoint,
)
from exp1.probes.heads import LinearProbeConfig, LinearProbeHead
from exp1.probes.losses import regression_loss, relative_depth_loss, surface_normal_loss
from exp1.probes.metrics import (
    regression_metrics,
    relative_depth_metrics,
    surface_normal_metrics,
    viewpoint_metrics,
)


REGRESSION_TASKS = {
    "camera_distance",
    "viewpoint",
    "lighting_direction",
    "lighting_intensity",
    "apparent_scale",
}


@dataclass
class ProbeTrainConfig:
    """Hyperparameters for frozen-feature linear probe training."""

    task: str = "surface_normal_aggregate"
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"
    use_layernorm: bool = True
    surface_normal_loss: str = "cosine"
    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 0.0


@dataclass
class ProbeArraySplit:
    """Numpy arrays and row metadata for one split."""

    features: np.ndarray
    targets: np.ndarray
    valid_mask: Optional[np.ndarray]
    render_ids: np.ndarray
    rows: pd.DataFrame


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
    return torch.device(requested)


def _as_2d_float(array: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    return arr


def _as_valid_mask(
    mask: Optional[Any], *, shape: tuple[int, int]
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape != shape:
        raise ValueError(f"valid_mask shape {arr.shape} does not match targets {shape}")
    return arr


def dataset_to_arrays(dataset: Exp1FeatureDataset) -> ProbeArraySplit:
    """Materialize an ``Exp1FeatureDataset`` into arrays for fast training."""
    features = []
    targets = []
    masks = []
    render_ids = []
    for index in range(len(dataset)):
        item = dataset[index]
        features.append(item["features"])
        targets.append(item["target"])
        render_ids.append(str(item["render_id"]))
        if item["valid_mask"] is not None:
            masks.append(item["valid_mask"])

    if not features:
        return ProbeArraySplit(
            features=np.zeros((0, 0), dtype=np.float32),
            targets=np.zeros((0, 0), dtype=np.float32),
            valid_mask=None,
            render_ids=np.asarray([], dtype=str),
            rows=dataset.rows.copy(),
        )

    target_arr = _as_2d_float(np.stack(targets, axis=0), name="targets")
    mask_arr = np.stack(masks, axis=0).astype(bool) if masks else None
    return ProbeArraySplit(
        features=_as_2d_float(np.stack(features, axis=0), name="features"),
        targets=target_arr,
        valid_mask=mask_arr,
        render_ids=np.asarray(render_ids, dtype=str),
        rows=dataset.rows.copy(),
    )


def _make_loader(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    valid_mask: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    x = torch.from_numpy(_as_2d_float(features, name="features"))
    y = torch.from_numpy(_as_2d_float(targets, name="targets"))
    mask = valid_mask
    if mask is None:
        mask = np.ones(y.shape, dtype=bool)
    m = torch.from_numpy(_as_valid_mask(mask, shape=y.shape).astype(bool))
    dataset = TensorDataset(x, y, m)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
    )


def _loss_for_task(
    *,
    task: str,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: ProbeTrainConfig,
) -> torch.Tensor:
    if task == "surface_normal_aggregate":
        return surface_normal_loss(
            outputs,
            targets,
            kind=str(cfg.surface_normal_loss),
        )
    if task == "relative_depth_regions":
        return relative_depth_loss(outputs, targets, valid_mask)
    if task == "lighting_direction":
        return surface_normal_loss(outputs, targets, kind="cosine")
    if task in REGRESSION_TASKS:
        return regression_loss(outputs, targets)
    raise ValueError(f"Unsupported Experiment 1 probe task: {task}")


def _metrics_for_task(
    *,
    task: str,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> dict[str, float]:
    if task == "surface_normal_aggregate":
        return surface_normal_metrics(outputs, targets)
    if task == "relative_depth_regions":
        if valid_mask is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
        return relative_depth_metrics(outputs, targets, valid_mask)
    if task == "lighting_direction":
        return surface_normal_metrics(outputs, targets)
    if task == "viewpoint":
        return viewpoint_metrics(outputs, targets)
    if task in REGRESSION_TASKS:
        return regression_metrics(outputs, targets)
    raise ValueError(f"Unsupported Experiment 1 probe task: {task}")


def evaluate_probe_arrays(
    model: torch.nn.Module,
    *,
    features: np.ndarray,
    targets: np.ndarray,
    valid_mask: Optional[np.ndarray],
    task: str,
    device: Union[str, torch.device],
    batch_size: int = 1024,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run probe inference and compute task metrics for numpy arrays."""
    device = _resolve_device(device)
    loader = _make_loader(
        features=features,
        targets=targets,
        valid_mask=valid_mask,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    model.eval()
    outputs = []
    target_chunks = []
    mask_chunks = []
    with torch.inference_mode():
        for xb, yb, mb in loader:
            logits = model(xb.to(device))
            outputs.append(logits.detach().cpu())
            target_chunks.append(yb)
            mask_chunks.append(mb)
    output_tensor = torch.cat(outputs, dim=0)
    target_tensor = torch.cat(target_chunks, dim=0)
    mask_tensor = torch.cat(mask_chunks, dim=0)
    metrics = _metrics_for_task(
        task=task,
        outputs=output_tensor,
        targets=target_tensor,
        valid_mask=mask_tensor,
    )
    return output_tensor, metrics


def _score(metrics: Mapping[str, float], *, task: str) -> float:
    if task == "surface_normal_aggregate":
        return float(metrics["angular_error_deg_mean"])
    if task == "relative_depth_regions":
        value = float(metrics["valid_pair_accuracy"])
        return value if np.isfinite(value) else -float("inf")
    if task == "lighting_direction":
        return float(metrics["angular_error_deg_mean"])
    if task == "viewpoint":
        return float(metrics["viewpoint_angular_error_deg_mean"])
    if task in REGRESSION_TASKS:
        return float(metrics["mae_mean"])
    raise ValueError(f"Unsupported Experiment 1 probe task: {task}")


def _is_better(
    value: float, best: Optional[float], *, task: str, min_delta: float
) -> bool:
    if best is None:
        return True
    if task == "surface_normal_aggregate":
        return value < best - float(min_delta)
    if task == "relative_depth_regions":
        return value > best + float(min_delta)
    if task in REGRESSION_TASKS:
        return value < best - float(min_delta)
    raise ValueError(f"Unsupported Experiment 1 probe task: {task}")


def train_probe_arrays(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    *,
    task: str = "surface_normal_aggregate",
    train_valid_mask: Optional[np.ndarray] = None,
    val_features: Optional[np.ndarray] = None,
    val_targets: Optional[np.ndarray] = None,
    val_valid_mask: Optional[np.ndarray] = None,
    config: Optional[ProbeTrainConfig] = None,
) -> dict[str, Any]:
    """Train a linear probe from in-memory arrays."""
    cfg = copy.deepcopy(config) if config is not None else ProbeTrainConfig(task=task)
    cfg.task = task
    _seed_everything(int(cfg.seed))
    device = _resolve_device(cfg.device)

    x_train = _as_2d_float(train_features, name="train_features")
    y_train = _as_2d_float(train_targets, name="train_targets")
    train_valid_mask = _as_valid_mask(train_valid_mask, shape=y_train.shape)
    if len(x_train) == 0:
        raise ValueError("Cannot train an Experiment 1 probe with an empty train split")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("train_features and train_targets have different row counts")

    has_val = val_features is not None and val_targets is not None
    x_val = y_val = mask_val = None
    if has_val:
        x_val = _as_2d_float(val_features, name="val_features")
        y_val = _as_2d_float(val_targets, name="val_targets")
        mask_val = _as_valid_mask(val_valid_mask, shape=y_val.shape)
        has_val = len(x_val) > 0

    probe_cfg = LinearProbeConfig(
        input_dim=int(x_train.shape[1]),
        output_dim=int(y_train.shape[1]),
        use_layernorm=bool(cfg.use_layernorm),
    )
    model = LinearProbeHead(probe_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    loader = _make_loader(
        features=x_train,
        targets=y_train,
        valid_mask=train_valid_mask,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        seed=int(cfg.seed),
    )

    history: list[dict[str, float]] = []
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_score: Optional[float] = None
    bad_epochs = 0

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        for xb, yb, mb in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb.to(device))
            loss = _loss_for_task(
                task=task,
                outputs=outputs,
                targets=yb.to(device),
                valid_mask=mb.to(device),
                cfg=cfg,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        _, train_metrics = evaluate_probe_arrays(
            model,
            features=x_train,
            targets=y_train,
            valid_mask=train_valid_mask,
            task=task,
            device=device,
            batch_size=int(cfg.batch_size),
        )
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        monitor_metrics = train_metrics
        if has_val and x_val is not None and y_val is not None:
            _, val_metrics = evaluate_probe_arrays(
                model,
                features=x_val,
                targets=y_val,
                valid_mask=mask_val,
                task=task,
                device=device,
                batch_size=int(cfg.batch_size),
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            monitor_metrics = val_metrics
        history.append(row)

        current_score = _score(monitor_metrics, task=task)
        if _is_better(
            current_score,
            best_score,
            task=task,
            min_delta=float(cfg.min_delta),
        ):
            best_score = current_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        if has_val and cfg.early_stopping and bad_epochs >= int(cfg.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    outputs: dict[str, Any] = {
        "model": model,
        "probe_config": probe_cfg,
        "train_config": cfg,
        "history": history,
    }
    train_predictions, train_metrics = evaluate_probe_arrays(
        model,
        features=x_train,
        targets=y_train,
        valid_mask=train_valid_mask,
        task=task,
        device=device,
        batch_size=int(cfg.batch_size),
    )
    outputs["metrics"] = {"train": train_metrics}
    outputs["predictions"] = {"train": train_predictions}
    if has_val and x_val is not None and y_val is not None:
        val_predictions, val_metrics = evaluate_probe_arrays(
            model,
            features=x_val,
            targets=y_val,
            valid_mask=mask_val,
            task=task,
            device=device,
            batch_size=int(cfg.batch_size),
        )
        outputs["metrics"]["val"] = val_metrics
        outputs["predictions"]["val"] = val_predictions
    return outputs


def _prediction_dataframe(
    *,
    split_name: str,
    task: str,
    render_ids: np.ndarray,
    predictions: torch.Tensor,
    targets: np.ndarray,
    valid_mask: Optional[np.ndarray],
    target_columns: Optional[Sequence[str]],
    row_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    pred = predictions.detach().cpu().numpy().astype(np.float32)
    target = _as_2d_float(targets, name="targets")
    rows: dict[str, Any] = {
        "render_id": np.asarray(render_ids, dtype=str),
        "split": split_name,
    }
    if row_metadata is not None:
        metadata = row_metadata.reset_index(drop=True)
        for column in ("object_id", "source_dataset", "category", "texture_condition"):
            if column in metadata.columns:
                rows[column] = metadata[column].to_numpy()
    if task == "surface_normal_aggregate":
        names = list(
            target_columns or ["mean_normal_x", "mean_normal_y", "mean_normal_z"]
        )
        for idx, name in enumerate(names):
            rows[f"pred_{name}"] = pred[:, idx]
            rows[f"target_{name}"] = target[:, idx]
        pred_norm = pred / np.clip(
            np.linalg.norm(pred, axis=1, keepdims=True), 1e-8, None
        )
        target_norm = target / np.clip(
            np.linalg.norm(target, axis=1, keepdims=True),
            1e-8,
            None,
        )
        cos = np.clip((pred_norm * target_norm).sum(axis=1), -1.0, 1.0)
        rows["angular_error_deg"] = np.degrees(np.arccos(cos))
        return pd.DataFrame(rows)

    if task == "relative_depth_regions":
        mask = _as_valid_mask(valid_mask, shape=target.shape)
        probs = torch.sigmoid(predictions).detach().cpu().numpy().astype(np.float32)
        pred_labels = (probs >= 0.5).astype(np.int32)
        correct = pred_labels == target.astype(np.int32)
        valid_counts = mask.sum(axis=1)
        rows["row_valid_pair_accuracy"] = np.divide(
            (correct & mask).sum(axis=1),
            valid_counts,
            out=np.full(mask.shape[0], np.nan, dtype=np.float32),
            where=valid_counts > 0,
        )
        rows["row_valid_pair_count"] = valid_counts.astype(np.float32)
        for idx in range(pred.shape[1]):
            rows[f"pair_{idx}_logit"] = pred[:, idx]
            rows[f"pair_{idx}_prob"] = probs[:, idx]
            rows[f"pair_{idx}_label"] = target[:, idx].astype(np.int32)
            rows[f"pair_{idx}_valid"] = mask[:, idx].astype(bool)
        return pd.DataFrame(rows)

    if task in REGRESSION_TASKS:
        names = list(
            target_columns or [f"target_{idx}" for idx in range(pred.shape[1])]
        )
        if len(names) != pred.shape[1]:
            raise ValueError(
                f"Expected {pred.shape[1]} target columns for {task}, got {len(names)}"
            )
        for idx, name in enumerate(names):
            rows[f"pred_{name}"] = pred[:, idx]
            rows[f"target_{name}"] = target[:, idx]
            rows[f"abs_error_{name}"] = np.abs(pred[:, idx] - target[:, idx])
        rows["row_mae"] = np.abs(pred - target).mean(axis=1)
        if task == "lighting_direction":
            pred_norm = pred / np.clip(
                np.linalg.norm(pred, axis=1, keepdims=True),
                1e-8,
                None,
            )
            target_norm = target / np.clip(
                np.linalg.norm(target, axis=1, keepdims=True),
                1e-8,
                None,
            )
            cos = np.clip((pred_norm * target_norm).sum(axis=1), -1.0, 1.0)
            rows["angular_error_deg"] = np.degrees(np.arccos(cos))
        if task == "viewpoint":
            pred_az = np.arctan2(pred[:, 0], pred[:, 1])
            target_az = np.arctan2(target[:, 0], target[:, 1])
            pred_el = np.arctan2(pred[:, 2], pred[:, 3])
            target_el = np.arctan2(target[:, 2], target[:, 3])
            az_error = np.degrees(
                np.abs(
                    np.arctan2(np.sin(pred_az - target_az), np.cos(pred_az - target_az))
                )
            )
            el_error = np.degrees(
                np.abs(
                    np.arctan2(np.sin(pred_el - target_el), np.cos(pred_el - target_el))
                )
            )
            rows["azimuth_angular_error_deg"] = az_error
            rows["elevation_angular_error_deg"] = el_error
            rows["viewpoint_angular_error_deg"] = 0.5 * (az_error + el_error)
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported Experiment 1 probe task: {task}")


def train_exp1_probe(
    *,
    feature_cache: Union[str, Path],
    label_path: Union[str, Path],
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    task: str,
    model_name: str,
    layer_name: str,
    target_columns: Optional[Sequence[str]] = None,
    texture_condition: Optional[Union[str, Sequence[str]]] = None,
    train_texture_condition: Optional[Union[str, Sequence[str]]] = None,
    eval_texture_condition: Optional[Union[str, Sequence[str]]] = None,
    config: Optional[ProbeTrainConfig] = None,
) -> dict[str, Any]:
    """Train one Experiment 1 probe and save checkpoint/metrics/predictions."""
    cfg = copy.deepcopy(config) if config is not None else ProbeTrainConfig(task=task)
    cfg.task = task
    manifest = load_manifest(manifest_path, validate=False)
    assert_object_disjoint_splits(manifest)

    split_arrays: dict[str, ProbeArraySplit] = {}
    train_texture = train_texture_condition or texture_condition
    eval_texture = eval_texture_condition or texture_condition
    for split_name in ("train", "val", "test"):
        split_texture = (
            train_texture if split_name in {"train", "val"} else eval_texture
        )
        dataset = Exp1FeatureDataset(
            feature_cache,
            label_path,
            manifest=manifest,
            task=task,
            target_columns=target_columns,
            split=split_name,
            texture_condition=split_texture,
            require_label_valid=True,
        )
        arrays = dataset_to_arrays(dataset)
        if len(arrays.render_ids) > 0:
            split_arrays[split_name] = arrays
    if "train" not in split_arrays:
        raise ValueError(
            "No train rows are available after feature/label/manifest joins"
        )

    train_split = split_arrays["train"]
    val_split = split_arrays.get("val")
    result = train_probe_arrays(
        train_split.features,
        train_split.targets,
        task=task,
        train_valid_mask=train_split.valid_mask,
        val_features=val_split.features if val_split is not None else None,
        val_targets=val_split.targets if val_split is not None else None,
        val_valid_mask=val_split.valid_mask if val_split is not None else None,
        config=cfg,
    )
    model = result["model"]

    metrics: dict[str, Any] = {}
    prediction_frames = []
    for split_name, arrays in split_arrays.items():
        predictions, split_metrics = evaluate_probe_arrays(
            model,
            features=arrays.features,
            targets=arrays.targets,
            valid_mask=arrays.valid_mask,
            task=task,
            device=cfg.device,
            batch_size=int(cfg.batch_size),
        )
        metrics[split_name] = split_metrics
        prediction_frames.append(
            _prediction_dataframe(
                split_name=split_name,
                task=task,
                render_ids=arrays.render_ids,
                predictions=predictions,
                targets=arrays.targets,
                valid_mask=arrays.valid_mask,
                target_columns=target_columns,
                row_metadata=arrays.rows,
            )
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metadata = {
        "task": task,
        "model_name": model_name,
        "layer_name": layer_name,
        "feature_cache": str(feature_cache),
        "label_path": str(label_path),
        "manifest_path": str(manifest_path),
        "texture_condition": texture_condition,
        "train_texture_condition": train_texture_condition,
        "eval_texture_condition": eval_texture_condition,
        "num_rows_by_split": {
            split_name: int(len(arrays.render_ids))
            for split_name, arrays in split_arrays.items()
        },
    }
    artifact_paths = {
        "checkpoint": save_probe_checkpoint(
            output_dir / "checkpoint.pt",
            model=model,
            probe_config=result["probe_config"],
            train_config=cfg,
            metrics=metrics,
            metadata=metadata,
        ),
        "metrics": save_metrics_json(
            output_dir / "metrics.json",
            {"metrics": metrics, "metadata": metadata},
        ),
        "history": save_history_csv(output_dir / "history.csv", result["history"]),
        "predictions": save_predictions_csv(
            output_dir / "predictions.csv",
            predictions_df,
        ),
    }
    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions_df,
        "history": result["history"],
        "metadata": metadata,
        "artifact_paths": artifact_paths,
        "train_config": asdict(cfg),
    }

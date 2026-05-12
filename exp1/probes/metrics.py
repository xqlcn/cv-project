"""Metrics for Experiment 1 probe outputs."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def angular_error_deg(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Row-wise angular error in degrees between predicted and target vectors."""
    pred = F.normalize(predictions.float(), dim=-1, eps=1e-8)
    target = F.normalize(targets.float(), dim=-1, eps=1e-8)
    cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def surface_normal_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    errors = angular_error_deg(predictions, targets)
    return {
        "angular_error_deg_mean": float(errors.mean().item()),
        "angular_error_deg_median": float(errors.median().item()),
    }


def regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """MAE/RMSE metrics for scalar or vector regression targets."""
    pred = predictions.float()
    target = targets.float()
    diff = pred - target
    abs_diff = diff.abs()
    mse_per_target = diff.pow(2).mean(dim=0)
    mae_per_target = abs_diff.mean(dim=0)
    metrics: Dict[str, float] = {
        "mae_mean": float(abs_diff.mean().item()),
        "mse_mean": float(diff.pow(2).mean().item()),
        "rmse_mean": float(torch.sqrt(diff.pow(2).mean()).item()),
    }
    for idx, value in enumerate(mae_per_target):
        metrics[f"target_{idx}_mae"] = float(value.item())
    for idx, value in enumerate(mse_per_target):
        metrics[f"target_{idx}_rmse"] = float(torch.sqrt(value).item())
    return metrics


def _circular_abs_error_deg(
    pred_rad: torch.Tensor, target_rad: torch.Tensor
) -> torch.Tensor:
    diff = torch.atan2(
        torch.sin(pred_rad - target_rad), torch.cos(pred_rad - target_rad)
    )
    return torch.rad2deg(diff.abs())


def viewpoint_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Angular errors for sin/cos azimuth/elevation regression outputs."""
    if predictions.shape[-1] != 4 or targets.shape[-1] != 4:
        raise ValueError(
            "viewpoint metrics expect columns "
            "[azimuth_sin, azimuth_cos, elevation_sin, elevation_cos]"
        )
    pred = predictions.float()
    target = targets.float()
    pred_az = torch.atan2(pred[:, 0], pred[:, 1])
    target_az = torch.atan2(target[:, 0], target[:, 1])
    pred_el = torch.atan2(pred[:, 2], pred[:, 3])
    target_el = torch.atan2(target[:, 2], target[:, 3])
    az_error = _circular_abs_error_deg(pred_az, target_az)
    el_error = _circular_abs_error_deg(pred_el, target_el)
    combined = 0.5 * (az_error + el_error)
    return {
        "azimuth_angular_error_deg_mean": float(az_error.mean().item()),
        "elevation_angular_error_deg_mean": float(el_error.mean().item()),
        "viewpoint_angular_error_deg_mean": float(combined.mean().item()),
    }


def _balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    recalls = []
    for cls in (0, 1):
        mask = labels == cls
        if mask.any():
            recalls.append(float((preds[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def relative_depth_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """Accuracy metrics over valid relative-depth pairs only."""
    valid = valid_mask.detach().cpu().numpy().astype(bool)
    labels = targets.detach().cpu().numpy().astype(np.int32)
    preds = (torch.sigmoid(logits.detach()).cpu().numpy() >= 0.5).astype(np.int32)

    if not valid.any():
        return {
            "valid_pair_accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "valid_pair_count": 0.0,
        }

    y_true = labels[valid]
    y_pred = preds[valid]
    metrics: Dict[str, float] = {
        "valid_pair_accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
        "valid_pair_count": float(valid.sum()),
    }
    for pair_idx in range(valid.shape[1]):
        pair_valid = valid[:, pair_idx]
        if pair_valid.any():
            metrics[f"pair_{pair_idx}_accuracy"] = float(
                (labels[pair_valid, pair_idx] == preds[pair_valid, pair_idx]).mean()
            )
    return metrics

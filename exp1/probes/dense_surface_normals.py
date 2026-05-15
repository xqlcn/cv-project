"""Dense surface-normal probe head, loss, and metrics for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DenseSurfaceNormalHeadConfig",
    "DenseSurfaceNormalHead",
    "dense_surface_normal_loss",
    "dense_surface_normal_metrics",
    "dense_surface_normal_sample_metrics",
]


@dataclass
class DenseSurfaceNormalHeadConfig:
    """Configuration for the dense surface-normal linear probe."""

    feature_dim: int
    use_layernorm: bool = True


class DenseSurfaceNormalHead(nn.Module):
    """1x1 linear head applied to patch tokens, producing xyz normal per patch."""

    def __init__(self, cfg: DenseSurfaceNormalHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        modules: list[nn.Module] = []
        if cfg.use_layernorm:
            modules.append(nn.LayerNorm(int(cfg.feature_dim)))
        modules.append(nn.Linear(int(cfg.feature_dim), 3))
        self.head = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``features``: ``(B, P, P, D)``. Returns ``(B, P, P, 3)``."""
        if features.ndim != 4:
            raise ValueError(
                "DenseSurfaceNormalHead expects (B,P,P,D), "
                f"got shape {tuple(features.shape)}"
            )
        b, p1, p2, d = features.shape
        x = features.reshape(b * p1 * p2, d)
        return self.head(x).reshape(b, p1, p2, 3)


def _masked_unit_vectors(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}"
        )
    if pred.ndim != 4 or pred.shape[-1] != 3:
        raise ValueError(f"Expected dense normals (B,P,P,3), got {tuple(pred.shape)}")
    if mask.shape != pred.shape[:-1]:
        raise ValueError(f"mask shape {tuple(mask.shape)} != normal grid {pred.shape[:-1]}")
    valid = mask.bool() & torch.isfinite(target).all(dim=-1)
    pred_unit = F.normalize(pred.float(), dim=-1, eps=1e-8)
    target_unit = F.normalize(target.float(), dim=-1, eps=1e-8)
    return pred_unit, target_unit, valid


def dense_surface_normal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked cosine loss over valid surface-normal patches."""
    pred_unit, target_unit, valid = _masked_unit_vectors(pred, target, mask)
    if not valid.any():
        return pred.sum() * 0.0
    cos = (pred_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos[valid]).mean()


def _angular_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_unit, target_unit, valid = _masked_unit_vectors(pred, target, mask)
    if not valid.any():
        return torch.empty(0, dtype=torch.float32), valid
    cos = (pred_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    errors = torch.rad2deg(torch.acos(cos))
    return errors[valid], valid


def dense_surface_normal_sample_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> list[dict[str, float]]:
    """Compute per-render dense-normal metrics for CSV logging/bootstrap."""
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    mask = mask.detach().cpu().bool()
    pred_unit, target_unit, valid = _masked_unit_vectors(pred, target, mask)
    cos = (pred_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    errors = torch.rad2deg(torch.acos(cos)).numpy().astype(np.float32)
    valid_np = valid.numpy().astype(bool)

    rows: list[dict[str, float]] = []
    for i in range(errors.shape[0]):
        sample_errors = errors[i][valid_np[i]]
        row = {
            "valid_patch_count": float(valid_np[i].sum()),
            "angular_error_deg_mean": float("nan"),
            "angular_error_deg_median": float("nan"),
            "within_11_25_deg": float("nan"),
            "within_22_5_deg": float("nan"),
            "within_30_deg": float("nan"),
        }
        if sample_errors.size:
            row["angular_error_deg_mean"] = float(sample_errors.mean())
            row["angular_error_deg_median"] = float(np.median(sample_errors))
            row["within_11_25_deg"] = float((sample_errors < 11.25).mean())
            row["within_22_5_deg"] = float((sample_errors < 22.5).mean())
            row["within_30_deg"] = float((sample_errors < 30.0).mean())
        rows.append(row)
    return rows


def dense_surface_normal_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Mapping[str, float]:
    """Aggregate dense-normal angular-error metrics across valid patches."""
    errors, valid = _angular_errors(pred, target, mask)
    if errors.numel() == 0:
        return {
            "angular_error_deg_mean": float("nan"),
            "angular_error_deg_median": float("nan"),
            "within_11_25_deg": float("nan"),
            "within_22_5_deg": float("nan"),
            "within_30_deg": float("nan"),
            "valid_patch_count_mean": 0.0,
            "num_samples": float(pred.shape[0]),
        }
    per_sample_counts = valid.reshape(valid.shape[0], -1).sum(dim=1).float()
    return {
        "angular_error_deg_mean": float(errors.mean().item()),
        "angular_error_deg_median": float(errors.median().item()),
        "within_11_25_deg": float((errors < 11.25).float().mean().item()),
        "within_22_5_deg": float((errors < 22.5).float().mean().item()),
        "within_30_deg": float((errors < 30.0).float().mean().item()),
        "valid_patch_count_mean": float(per_sample_counts.mean().item()),
        "num_samples": float(pred.shape[0]),
    }

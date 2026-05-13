"""Linear dense-depth probe head, loss, and metrics for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "DenseDepthHeadConfig",
    "DenseDepthHead",
    "ssi_l1_loss",
    "dense_depth_metrics",
    "dense_depth_sample_metrics",
    "match_scale_and_shift",
]


@dataclass
class DenseDepthHeadConfig:
    """Configuration for the dense-depth linear probe."""

    feature_dim: int
    use_layernorm: bool = True


class DenseDepthHead(nn.Module):
    """1x1 linear head applied to patch tokens, producing one scalar per patch."""

    def __init__(self, cfg: DenseDepthHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        modules: list[nn.Module] = []
        if cfg.use_layernorm:
            modules.append(nn.LayerNorm(int(cfg.feature_dim)))
        modules.append(nn.Linear(int(cfg.feature_dim), 1))
        self.head = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``features``: ``(B, P, P, D)``. Returns ``(B, P, P)`` predictions."""
        if features.ndim != 4:
            raise ValueError(
                f"DenseDepthHead expects (B,P,P,D), got shape {tuple(features.shape)}"
            )
        b, p1, p2, d = features.shape
        x = features.reshape(b * p1 * p2, d)
        y = self.head(x).reshape(b, p1, p2)
        return y


def _ssi_normalize(
    values: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Subtract median and divide by mean absolute deviation per sample.

    Following the MiDaS recipe: ``shift = median(d)``, ``scale = mean(|d - shift|)``.
    Produces a per-sample affine-normalized tensor with zero median and unit
    average deviation, computed only over valid pixels.
    """
    if values.shape != mask.shape:
        raise ValueError(
            f"values shape {tuple(values.shape)} != mask shape {tuple(mask.shape)}"
        )
    out = torch.zeros_like(values)
    flat = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    for i in range(values.shape[0]):
        m = flat_mask[i]
        if m.any():
            v = flat[i][m]
            shift = torch.median(v)
            scale = torch.mean(torch.abs(v - shift))
            denom = scale if scale > eps else torch.full_like(scale, eps)
            normalized = (flat[i] - shift) / denom
            normalized = torch.where(m, normalized, torch.zeros_like(normalized))
            out[i] = normalized.reshape(out[i].shape)
    return out


def ssi_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Scale- and shift-invariant L1 loss for dense depth.

    Both ``pred`` and ``target`` are first per-sample affine-normalized using
    their valid pixels, then a masked L1 is computed.
    """
    pred = pred.float()
    target = target.float()
    mask = mask.bool()
    pred_norm = _ssi_normalize(pred, mask)
    target_norm = _ssi_normalize(target, mask)
    diff = torch.abs(pred_norm - target_norm) * mask.float()
    denom = torch.clamp(mask.float().sum(), min=1.0)
    return diff.sum() / denom


def _ssi_normalize_numpy(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1).astype(np.float32)
    flat_mask = mask.reshape(mask.shape[0], -1).astype(bool)
    for i in range(values.shape[0]):
        m = flat_mask[i]
        if not m.any():
            continue
        v = flat[i][m]
        shift = float(np.median(v))
        scale = float(np.mean(np.abs(v - shift)))
        denom = scale if scale > 1e-6 else 1e-6
        normalized = (flat[i] - shift) / denom
        normalized = np.where(m, normalized, 0.0)
        out[i] = normalized.reshape(out[i].shape)
    return out


def match_scale_and_shift(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Least-squares align ``prediction`` to ``target`` per sample."""
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    out = np.array(pred, copy=True, dtype=np.float32)
    for i in range(pred.shape[0]):
        m = valid[i]
        if int(m.sum()) < 2:
            continue
        p = pred[i][m]
        t = tgt[i][m]
        design = np.stack([p, np.ones_like(p)], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(design, t, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[i] = pred[i] * float(coef[0]) + float(coef[1])
    return out


def _depth_metrics_for_values(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    positive = target > 1e-6
    if not positive.any():
        return {
            "abs_rel": float("nan"),
            "rmse": float("nan"),
            "rmse_log": float("nan"),
            "d1": float("nan"),
            "d2": float("nan"),
            "d3": float("nan"),
        }
    tgt = target[positive].astype(np.float32)
    pr = np.clip(pred[positive].astype(np.float32), 1e-6, None)
    ratio = np.maximum(pr / tgt, tgt / pr)
    return {
        "abs_rel": float(np.mean(np.abs(pr - tgt) / tgt)),
        "rmse": float(np.sqrt(np.mean((pr - tgt) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pr) - np.log(tgt)) ** 2))),
        "d1": float(np.mean(ratio < 1.25)),
        "d2": float(np.mean(ratio < 1.25 ** 2)),
        "d3": float(np.mean(ratio < 1.25 ** 3)),
    }


def dense_depth_sample_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> list[dict[str, float]]:
    """Compute per-render dense-depth metrics for CSV logging/bootstrap."""
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    target_np = target.detach().cpu().numpy().astype(np.float32)
    mask_np = mask.detach().cpu().numpy().astype(bool)

    pred_ssi = _ssi_normalize_numpy(pred_np, mask_np)
    target_ssi = _ssi_normalize_numpy(target_np, mask_np)
    pred_si = match_scale_and_shift(pred_np, target_np, mask_np)

    rows: list[dict[str, float]] = []
    for i in range(pred_np.shape[0]):
        m = mask_np[i]
        valid_count = int(m.sum())
        row: dict[str, float] = {
            "valid_patch_count": float(valid_count),
            "ssi_l1": float("nan"),
            "pearson_r": float("nan"),
        }
        if valid_count == 0:
            rows.append(row)
            continue
        p = pred_np[i][m]
        t = target_np[i][m]
        p_ssi = pred_ssi[i][m]
        t_ssi = target_ssi[i][m]
        row["ssi_l1"] = float(np.mean(np.abs(p_ssi - t_ssi)))
        if p.size > 1 and np.std(p) > 0 and np.std(t) > 0:
            row["pearson_r"] = float(np.corrcoef(p, t)[0, 1])

        for prefix, values in (
            ("scale_aware", p),
            ("scale_invariant", pred_si[i][m]),
        ):
            metrics = _depth_metrics_for_values(values, t)
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def dense_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Mapping[str, float]:
    """Compute Probe3D-style scale-aware and scale-invariant depth metrics."""
    sample_rows = dense_depth_sample_metrics(pred, target, mask)

    def _values(column: str) -> list[float]:
        values = []
        for row in sample_rows:
            value = float(row.get(column, float("nan")))
            if np.isfinite(value):
                values.append(value)
        return values

    def _agg(values: list[float], stat: str = "mean") -> float:
        if not values:
            return float("nan")
        if stat == "mean":
            return float(np.mean(values))
        if stat == "median":
            return float(np.median(values))
        raise ValueError(f"Unsupported stat {stat}")

    metrics: dict[str, float] = {
        "ssi_l1_mean": _agg(_values("ssi_l1")),
        "ssi_l1_median": _agg(_values("ssi_l1"), "median"),
        "pearson_r_mean": _agg(_values("pearson_r")),
        "valid_patch_count_mean": _agg(_values("valid_patch_count")),
    }
    for prefix in ("scale_aware", "scale_invariant"):
        for metric in ("abs_rel", "rmse", "rmse_log", "d1", "d2", "d3"):
            metrics[f"{prefix}_{metric}_mean"] = _agg(
                _values(f"{prefix}_{metric}")
            )

    # Backward-compatible aliases used by earlier dense-depth tests/scripts.
    metrics["abs_rel_mean"] = metrics["scale_invariant_abs_rel_mean"]
    metrics["abs_rel_median"] = _agg(_values("scale_invariant_abs_rel"), "median")
    metrics["rmse_log_mean"] = metrics["scale_invariant_rmse_log_mean"]
    metrics["delta_1_mean"] = metrics["scale_invariant_d1_mean"]
    metrics["delta_2_mean"] = metrics["scale_invariant_d2_mean"]
    metrics["delta_3_mean"] = metrics["scale_invariant_d3_mean"]
    metrics["num_samples"] = float(len(_values("scale_invariant_abs_rel")))
    return metrics

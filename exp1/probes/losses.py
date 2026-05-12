"""Loss functions for Experiment 1 linear probes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def surface_normal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    kind: str = "cosine",
) -> torch.Tensor:
    """Loss for aggregate surface-normal regression."""
    pred = F.normalize(predictions.float(), dim=-1, eps=1e-8)
    target = F.normalize(targets.float(), dim=-1, eps=1e-8)
    if kind == "cosine":
        return (1.0 - (pred * target).sum(dim=-1)).mean()
    if kind == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported surface normal loss: {kind}")


def relative_depth_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked BCE-with-logits for relative-depth region pairs."""
    valid = valid_mask.bool()
    if not valid.any():
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
    )
    return losses[valid].mean()


def regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared error for scalar/vector regression probes."""
    return F.mse_loss(predictions.float(), targets.float())

"""Loss helpers for probe training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def probe_loss_binary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    logistic: bool,
) -> torch.Tensor:
    if logistic:
        return F.binary_cross_entropy_with_logits(logits.view(-1), targets.float().view(-1))
    return F.cross_entropy(logits, targets.long())


def probe_loss_regression(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(preds.view(-1), targets.float().view(-1))

"""Classification and regression metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score


def binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    logits: shape [N] (single logit) or [N,2]. labels: shape [N] int {0,1}.
    """
    if logits.ndim == 2 and logits.shape[1] == 2:
        prob = F.softmax(logits, dim=1)[:, 1]
        logit_for_auc = logits[:, 1] - logits[:, 0]
    else:
        logit_vec = logits.view(-1)
        prob = torch.sigmoid(logit_vec)
        logit_for_auc = logit_vec

    y_true = labels.detach().cpu().numpy().astype(np.int32)
    y_prob = prob.detach().cpu().numpy()
    y_pred = (y_prob >= threshold).astype(np.int32)

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    y_hat = preds.detach().cpu().numpy().reshape(-1)
    y = targets.detach().cpu().numpy().reshape(-1)
    mae = mean_absolute_error(y, y_hat)
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    r2 = r2_score(y, y_hat)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def cosine_similarity_pairs(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise cosine similarity between matching rows of a and b."""
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    return (a_n * b_n).sum(dim=-1)

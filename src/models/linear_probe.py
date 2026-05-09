"""Lightweight linear probes for frozen representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


TaskType = Literal["binary_classification", "multiclass", "regression"]


@dataclass
class ProbeConfig:
    input_dim: int
    num_outputs: int
    task: TaskType = "binary_classification"
    use_layernorm: bool = True
    logistic: bool = True  # binary: single logit + BCE; if False, use num_outputs=2 + CE


class LinearProbe(nn.Module):
    """
    Single linear layer (optionally preceded by LayerNorm).

    binary + logistic: num_outputs ignored (uses 1 output + BCEWithLogits).
    binary + not logistic: num_outputs should be 2 for softmax CE.
    """

    def __init__(self, cfg: ProbeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm: Optional[nn.LayerNorm] = None
        if cfg.use_layernorm:
            self.norm = nn.LayerNorm(cfg.input_dim)

        if cfg.task == "binary_classification" and cfg.logistic:
            out_dim = 1
        else:
            out_dim = cfg.num_outputs
        self.head = nn.Linear(cfg.input_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        return self.head(x)

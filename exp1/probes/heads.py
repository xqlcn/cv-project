"""Probe heads for cached Experiment 1 features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LinearProbeConfig:
    input_dim: int
    output_dim: int
    use_layernorm: bool = True


class LinearProbeHead(nn.Module):
    """A frozen-feature linear probe with optional LayerNorm."""

    def __init__(self, cfg: LinearProbeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm: Optional[nn.LayerNorm] = None
        if cfg.use_layernorm:
            self.norm = nn.LayerNorm(int(cfg.input_dim))
        self.head = nn.Linear(int(cfg.input_dim), int(cfg.output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            features = self.norm(features)
        return self.head(features)

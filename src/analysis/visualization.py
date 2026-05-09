"""Lightweight plotting helpers for probe metrics and similarity histograms."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torch


def plot_histogram(
    values: Iterable[float],
    *,
    title: str,
    out_path: Optional[Path] = None,
    bins: int = 40,
) -> None:
    xs = torch.tensor(list(values), dtype=torch.float32).numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(xs, bins=bins, color="#4C72B0", alpha=0.9)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.close()

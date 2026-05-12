"""Checkpoint and artifact helpers for Experiment 1 probes."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np
import pandas as pd
import torch


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_metrics_json(path: Union[str, Path], metrics: Mapping[str, Any]) -> Path:
    """Save nested metrics as stable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def save_history_csv(path: Union[str, Path], history: list[Mapping[str, Any]]) -> Path:
    """Save per-epoch probe training history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)
    return path


def save_predictions_csv(
    path: Union[str, Path],
    predictions: pd.DataFrame,
) -> Path:
    """Save per-render probe predictions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path


def save_probe_checkpoint(
    path: Union[str, Path],
    *,
    model: torch.nn.Module,
    probe_config: Any,
    train_config: Any,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    """Save a lightweight linear-probe checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "probe_config": _jsonable(probe_config),
            "train_config": _jsonable(train_config),
            "metrics": _jsonable(metrics),
            "metadata": _jsonable(metadata),
        },
        path,
    )
    return path

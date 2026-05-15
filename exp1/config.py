"""Small config helpers shared by Experiment 1 scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, cast

from omegaconf import DictConfig, OmegaConf


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_exp1_config_path() -> Path:
    return project_root() / "configs" / "exp1_smoke.yaml"


def _load_config_with_defaults(path: Path, *, seen: Optional[set[Path]] = None) -> Any:
    """Load a config file and recursively expand its simple defaults stack."""
    cfg_dir = path.resolve().parent
    resolved_path = path.resolve()
    seen = set() if seen is None else set(seen)
    if resolved_path in seen:
        raise ValueError(f"Cyclic config defaults include {resolved_path}")
    seen.add(resolved_path)
    raw_cfg = OmegaConf.load(path)
    if not OmegaConf.is_dict(raw_cfg):
        raise TypeError(
            f"Expected mapping config at {path}, got {type(raw_cfg).__name__}"
        )

    cfg = cast(DictConfig, raw_cfg)
    defaults_node = OmegaConf.select(cfg, "defaults", default=[])
    defaults = list(defaults_node) if defaults_node is not None else []
    parts: List[Any] = []

    for entry in defaults:
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            parts.append(
                _load_config_with_defaults(
                    cfg_dir / f"{entry}.yaml",
                    seen=seen,
                )
            )
        elif isinstance(entry, dict):
            for group, name in entry.items():
                if name in {None, "null"}:
                    continue
                parts.append(
                    _load_config_with_defaults(
                        cfg_dir / str(group) / f"{name}.yaml",
                        seen=seen,
                    )
                )
        else:
            raise TypeError(f"Unsupported defaults entry in {path}: {entry!r}")

    parts.append(cfg)
    return OmegaConf.merge(*parts)


def load_exp1_config(path: Path) -> DictConfig:
    """Load an Experiment 1 config with its simple Hydra-style defaults stack."""
    OmegaConf.register_new_resolver("now", lambda fmt: "now", replace=True)
    merged = _load_config_with_defaults(path)
    if not OmegaConf.is_dict(merged):
        raise TypeError(
            f"Merged config is not a mapping for {path}: {type(merged).__name__}"
        )
    OmegaConf.resolve(merged)
    return cast(DictConfig, merged)


def resolve_path(base: Path, path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    out = Path(path)
    return out if out.is_absolute() else base / out


def ensure_local_hf_home(cfg: DictConfig) -> Path:
    """Default Hugging Face caches to the project data directory.

    This keeps CLIP/DINOv2/ShapeNet downloads out of user-level cache paths that
    may be unavailable inside sandboxed runs, while preserving an explicit
    caller-provided ``HF_HOME``.
    """
    root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    configured = OmegaConf.select(cfg, "paths.hf_cache_root", default="data/hf_cache")
    path = resolve_path(root, str(configured))
    assert path is not None
    os.environ.setdefault("HF_HOME", str(path))
    return path

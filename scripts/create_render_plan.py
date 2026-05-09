#!/usr/bin/env python3
"""Create an Experiment 1 render plan from a lightweight asset manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, cast

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.metadata.manifest import save_manifest
from exp1.rendering.grid import build_render_plan_from_config, load_asset_manifest


def _default_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "exp1_smoke.yaml"


def _load_exp1_config(path: Path) -> DictConfig:
    """Load the small Experiment 1 defaults stack without requiring Hydra runtime."""
    OmegaConf.register_new_resolver("now", lambda fmt: "now", replace=True)
    cfg_dir = path.resolve().parent
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
            parts.append(OmegaConf.load(cfg_dir / f"{entry}.yaml"))
        elif isinstance(entry, dict):
            for group, name in entry.items():
                if name in {None, "null"}:
                    continue
                parts.append(OmegaConf.load(cfg_dir / str(group) / f"{name}.yaml"))
        else:
            raise TypeError(f"Unsupported defaults entry in {path}: {entry!r}")

    parts.append(cfg)
    merged = OmegaConf.merge(*parts)
    if not OmegaConf.is_dict(merged):
        raise TypeError(
            f"Merged config is not a mapping for {path}: {type(merged).__name__}"
        )
    OmegaConf.resolve(merged)
    return cast(DictConfig, merged)


def _resolve_path(project_root: Path, path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    out = Path(path)
    return out if out.is_absolute() else project_root / out


def _first_enabled_manifest(cfg: DictConfig) -> Path:
    for source in cfg.assets.sources:
        has_manifest = source.get("manifest_path") is not None
        if bool(source.get("enabled", False)) and has_manifest:
            return Path(str(source.manifest_path))
    raise ValueError("No enabled asset source with a manifest_path was found in config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=None,
        help="Optional synthetic/ModelNet asset manifest. Defaults to config source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output render manifest path. Defaults to paths.render_plan_jsonl.",
    )
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Write the plan without schema validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()

    asset_manifest = args.asset_manifest or _first_enabled_manifest(cfg)
    asset_manifest = _resolve_path(project_root, str(asset_manifest))
    if asset_manifest is None or not asset_manifest.is_file():
        raise FileNotFoundError(
            f"Missing asset manifest: {asset_manifest}. "
            "Run scripts/setup_synthetic_primitives.py or pass --asset-manifest."
        )

    output = args.output
    if output is None:
        output = Path(str(cfg.paths.render_plan_jsonl))
    output = _resolve_path(project_root, str(output))
    assert output is not None

    asset_rows = load_asset_manifest(asset_manifest)
    plan = build_render_plan_from_config(
        asset_rows,
        cfg,
        max_objects=args.max_objects,
    )
    save_manifest(plan, output, validate=not args.no_validate)

    print(f"Wrote {len(plan)} render rows to {output}")
    print(f"Objects: {plan['object_id'].nunique()}")
    textures = ", ".join(sorted(plan["texture_condition"].unique()))
    print(f"Texture conditions: {textures}")


if __name__ == "__main__":
    main()

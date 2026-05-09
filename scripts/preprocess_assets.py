#!/usr/bin/env python3
"""Discover and normalize Experiment 1 mesh assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.assets.discover import (
    discover_assets_from_directory,
    discover_modelnet40_assets,
    load_assets_from_manifest,
)
from exp1.assets.normalize import normalize_asset_manifest
from exp1.assets.validate import validate_asset_manifest, write_object_split_manifest
from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from src.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--modelnet-root", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--normalized-manifest", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--normalized-root", type=Path, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--target-extent", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Only discover assets and write the raw asset manifest.",
    )
    return parser.parse_args()


def _path_from_cfg(
    project_root: Path,
    explicit: Optional[Path],
    configured: str,
) -> Path:
    path = str(explicit) if explicit is not None else configured
    resolved = resolve_path(project_root, path)
    assert resolved is not None
    return resolved


def _cli_source(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Return the first explicit CLI source, preserving the historical priority."""
    if args.input_manifest is not None:
        return {
            "name": "input_manifest",
            "type": "manifest",
            "manifest_path": str(args.input_manifest),
        }
    if args.modelnet_root is not None:
        return {
            "name": "modelnet40_cli",
            "type": "modelnet40",
            "root": str(args.modelnet_root),
        }
    if args.asset_root is not None:
        return {
            "name": "asset_root",
            "type": "mesh_directory",
            "root": str(args.asset_root),
            "source_dataset": "custom_assets",
        }
    return None


def _source_max_objects(
    global_max: Optional[int],
    source: Mapping[str, Any],
) -> Optional[int]:
    if global_max is not None:
        return global_max
    value = source.get("max_objects")
    return int(value) if value is not None else None


def _discover_from_source(
    source: Mapping[str, Any],
    *,
    project_root: Path,
    allowed_extensions: Iterable[str],
    max_objects: Optional[int],
) -> List[Dict[str, Any]]:
    source_type = str(source.get("type", "manifest"))
    source_dataset = str(source.get("source_dataset", source.get("name", "unknown")))

    if source_type in {"manifest", "synthetic_manifest", "asset_manifest"}:
        manifest_value = source.get("manifest_path")
        if manifest_value is None:
            raise ValueError(f"Source {source.get('name')} is missing manifest_path")
        manifest_path = resolve_path(project_root, str(manifest_value))
        assert manifest_path is not None
        return load_assets_from_manifest(
            manifest_path,
            source_dataset=source_dataset,
            max_objects=max_objects,
        )

    if source_type == "modelnet40":
        root_value = source.get("root")
        if root_value is None:
            raise ValueError(f"Source {source.get('name')} is missing root")
        root = resolve_path(project_root, str(root_value))
        assert root is not None
        return discover_modelnet40_assets(root, max_objects=max_objects)

    if source_type == "mesh_directory":
        root_value = source.get("root")
        if root_value is None:
            raise ValueError(f"Source {source.get('name')} is missing root")
        root = resolve_path(project_root, str(root_value))
        if root is None:
            raise ValueError(f"Source {source.get('name')} is missing root")
        return discover_assets_from_directory(
            root,
            source_dataset=source_dataset,
            allowed_extensions=tuple(allowed_extensions),
            max_objects=max_objects,
        )

    raise ValueError(f"Unsupported asset source type: {source_type}")


def _discover_assets(args: argparse.Namespace, cfg: DictConfig) -> List[Dict[str, Any]]:
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    max_objects = args.max_objects
    allowed = tuple(str(ext).lower() for ext in cfg.assets.allowed_extensions)

    cli_source = _cli_source(args)
    if cli_source is not None:
        rows = _discover_from_source(
            cli_source,
            project_root=project_root,
            allowed_extensions=allowed,
            max_objects=max_objects,
        )
        if not rows:
            source_name = str(cli_source.get("name", "CLI source"))
            raise RuntimeError(
                f"No assets discovered from {source_name}. Check that the input path "
                "exists and contains supported mesh files."
            )
        return rows

    rows: List[Dict[str, Any]] = []
    for source in cfg.assets.sources:
        if not bool(source.get("enabled", False)):
            continue
        rows.extend(
            _discover_from_source(
                OmegaConf.to_container(source, resolve=True),
                project_root=project_root,
                allowed_extensions=allowed,
                max_objects=_source_max_objects(max_objects, source),
            )
        )
    if not rows:
        raise RuntimeError(
            "No assets discovered. Pass --input-manifest, --asset-root, "
            "--modelnet-root, or enable an asset source in the config."
        )
    if max_objects is not None:
        rows = rows[: int(max_objects)]
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()

    raw_manifest = _path_from_cfg(
        project_root,
        args.output_manifest,
        str(cfg.paths.asset_manifest),
    )
    normalized_manifest = _path_from_cfg(
        project_root,
        args.normalized_manifest,
        str(cfg.paths.normalized_asset_manifest),
    )
    split_manifest = _path_from_cfg(
        project_root,
        args.split_manifest,
        str(cfg.paths.split_manifest),
    )
    normalized_root = _path_from_cfg(
        project_root,
        args.normalized_root,
        str(Path(str(cfg.paths.exp1_root)) / "normalized_assets"),
    )

    discovered = _discover_assets(args, cfg)
    write_jsonl(raw_manifest, discovered)

    if args.no_normalize:
        print(f"Wrote {len(discovered)} discovered assets to {raw_manifest}")
        return

    normalized = normalize_asset_manifest(
        discovered,
        output_root=normalized_root,
        target_extent=float(args.target_extent),
        overwrite=bool(args.overwrite),
        fail_fast=bool(args.fail_fast),
    )
    validate_asset_manifest(normalized, require_normalized_paths=True)
    write_jsonl(normalized_manifest, normalized)
    write_object_split_manifest(normalized, split_manifest)

    failed = [row for row in normalized if row["asset_status"] == "failed"]
    print(f"Wrote raw asset manifest: {raw_manifest}")
    print(f"Wrote normalized asset manifest: {normalized_manifest}")
    print(f"Wrote object split manifest: {split_manifest}")
    print(f"Assets: {len(normalized)} total, {len(failed)} failed")


if __name__ == "__main__":
    main()

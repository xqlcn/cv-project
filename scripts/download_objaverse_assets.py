#!/usr/bin/env python3
"""Download a bounded Objaverse subset and write an Experiment 1 asset manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.assets.objaverse import bytes_from_gb, download_objaverse_assets
from exp1.config import (
    default_exp1_config_path,
    ensure_local_hf_home,
    load_exp1_config,
    resolve_path,
)
from src.utils.io import write_jsonl


def _resolve_required(project_root: Path, path: Any) -> Path:
    resolved = resolve_path(project_root, str(path))
    assert resolved is not None
    return resolved


def _optional_list(value: Optional[Any]):
    if value is None:
        return None
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = [str(part).strip() for part in value]
    return [value for value in values if value] or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--category", dest="categories", action="append", default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--max-download-gb", type=float, default=None)
    parser.add_argument("--download-processes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    ensure_local_hf_home(cfg)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    obj_cfg = cfg.get("datasets", {}).get("objaverse", {})

    output = (
        args.output
        if args.output is not None
        else _resolve_required(project_root, obj_cfg.get("manifest_path"))
    )
    cache_dir = (
        args.cache_dir
        if args.cache_dir is not None
        else _resolve_required(project_root, obj_cfg.get("cache_dir"))
    )
    if not output.is_absolute():
        output = project_root / output
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    os.environ.setdefault("OBJAVERSE_HOME", str(cache_dir))

    max_objects = int(args.max_objects or obj_cfg.get("max_objects", 100))
    max_download_gb = (
        args.max_download_gb
        if args.max_download_gb is not None
        else obj_cfg.get("max_download_gb")
    )
    categories = args.categories or _optional_list(obj_cfg.get("categories"))

    rows = download_objaverse_assets(
        cache_dir=cache_dir,
        categories=categories,
        max_objects=max_objects,
        max_download_bytes=bytes_from_gb(max_download_gb),
        download_processes=int(
            args.download_processes or obj_cfg.get("download_processes", 4)
        ),
        batch_size=int(args.batch_size or obj_cfg.get("batch_size", 8)),
        split=str(args.split or obj_cfg.get("split", "train")),
    )
    write_jsonl(output, rows)

    summary = {
        "output": str(output),
        "cache_dir": str(cache_dir),
        "categories": categories,
        "max_objects": max_objects,
        "max_download_gb": max_download_gb,
        "objects_written": len(rows),
        "manifest_bytes": sum(int(row.get("objaverse_cache_bytes", 0)) for row in rows),
    }
    print(OmegaConf.to_yaml(summary))


if __name__ == "__main__":
    main()

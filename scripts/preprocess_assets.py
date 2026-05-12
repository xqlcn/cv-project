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
from exp1.assets.shapenet import (
    DEFAULT_SHAPENET_HF_REPO_ID,
    discover_huggingface_shapenet_assets,
)
from exp1.assets.validate import (
    assign_object_disjoint_splits,
    validate_asset_manifest,
    write_object_split_manifest,
)
from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from src.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument(
        "--objaverse-root",
        type=Path,
        default=None,
        help="Local Objaverse mesh directory, scanned recursively.",
    )
    parser.add_argument(
        "--modelnet-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--shapenet-hf-repo-id",
        type=str,
        default=None,
        help=(
            "Hugging Face dataset repo. Defaults to the original "
            "ShapeNet/ShapeNetCore ZIP repository."
        ),
    )
    parser.add_argument("--shapenet-hf-local-dir", type=Path, default=None)
    parser.add_argument("--shapenet-hf-revision", type=str, default=None)
    parser.add_argument(
        "--shapenet-category",
        dest="shapenet_categories",
        action="append",
        default=None,
        help=(
            "Limit Hugging Face ShapeNet preprocessing to a category name or "
            "synset id. Repeat for multiple categories."
        ),
    )
    parser.add_argument(
        "--shapenet-no-download",
        action="store_true",
        help="Scan --shapenet-hf-local-dir without contacting Hugging Face.",
    )
    parser.add_argument(
        "--shapenet-no-extract-archives",
        action="store_true",
        help="Do not extract ShapeNetCore ZIP snapshots before scanning.",
    )
    parser.add_argument("--shapenet-extracted-dir", type=Path, default=None)
    parser.add_argument("--shapenet-overwrite-extract", action="store_true")
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


def _cli_sources(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Return explicit CLI sources.

    If an input manifest is supplied, it is treated as the complete source.
    Otherwise ShapeNetCore and Objaverse CLI sources can be combined in one run.
    """
    if args.input_manifest is not None:
        return [
            {
                "name": "input_manifest",
                "type": "manifest",
                "manifest_path": str(args.input_manifest),
            }
        ]
    sources: List[Dict[str, Any]] = []
    if args.shapenet_hf_repo_id is not None or args.shapenet_hf_local_dir is not None:
        sources.append(
            {
                "name": "shapenet_hf_cli",
                "type": "huggingface_shapenet",
                "repo_id": args.shapenet_hf_repo_id or DEFAULT_SHAPENET_HF_REPO_ID,
                "local_dir": (
                    str(args.shapenet_hf_local_dir)
                    if args.shapenet_hf_local_dir is not None
                    else None
                ),
                "revision": args.shapenet_hf_revision,
                "categories": args.shapenet_categories,
                "download": not args.shapenet_no_download,
                "extract_archives": not args.shapenet_no_extract_archives,
                "extracted_dir": (
                    str(args.shapenet_extracted_dir)
                    if args.shapenet_extracted_dir is not None
                    else None
                ),
                "overwrite_extract": args.shapenet_overwrite_extract,
                "source_dataset": "shapenet",
            }
        )
    if args.objaverse_root is not None:
        sources.append(
            {
                "name": "objaverse_cli",
                "type": "objaverse",
                "root": str(args.objaverse_root),
                "source_dataset": "objaverse",
                "skip_missing": True,
            }
        )
    if args.asset_root is not None:
        sources.append(
            {
                "name": "asset_root",
                "type": "mesh_directory",
                "root": str(args.asset_root),
                "source_dataset": "custom_assets",
            }
        )
    if args.modelnet_root is not None:
        sources.append(
            {
                "name": "modelnet40_cli",
                "type": "modelnet40",
                "root": str(args.modelnet_root),
            }
        )
    return sources


def _source_max_objects(
    global_max: Optional[int],
    source: Mapping[str, Any],
) -> Optional[int]:
    if global_max is not None:
        return global_max
    value = source.get("max_objects")
    return int(value) if value is not None else None


def _optional_path(
    project_root: Path,
    value: Optional[Any],
) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    return resolve_path(project_root, str(value))


def _optional_string_list(value: Optional[Any]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = [str(part).strip() for part in value]
    out = [part for part in values if part]
    return out or None


def _optional_bool(value: Optional[Any], *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(value)


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

    if source_type in {"huggingface_shapenet", "shapenet_hf"}:
        local_dir = _optional_path(project_root, source.get("local_dir"))
        extracted_dir = _optional_path(project_root, source.get("extracted_dir"))
        return discover_huggingface_shapenet_assets(
            repo_id=str(source.get("repo_id", DEFAULT_SHAPENET_HF_REPO_ID)),
            local_dir=local_dir,
            download=_optional_bool(source.get("download"), default=True),
            revision=source.get("revision"),
            token=source.get("token", True),
            categories=_optional_string_list(source.get("categories")),
            allow_patterns=_optional_string_list(source.get("allow_patterns")),
            ignore_patterns=_optional_string_list(source.get("ignore_patterns")),
            extract_archives=_optional_bool(
                source.get("extract_archives"),
                default=True,
            ),
            extracted_dir=extracted_dir,
            overwrite_extract=_optional_bool(
                source.get("overwrite_extract"),
                default=False,
            ),
            source_dataset=source_dataset,
            allowed_extensions=tuple(allowed_extensions),
            max_objects=max_objects,
        )

    if source_type in {"mesh_directory", "objaverse"}:
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


def _interleave_source_rows(
    sources: List[List[Dict[str, Any]]],
    *,
    max_objects: Optional[int],
) -> List[Dict[str, Any]]:
    if max_objects is None:
        return [row for rows in sources for row in rows]

    selected: List[Dict[str, Any]] = []
    indexes = [0 for _ in sources]
    while len(selected) < int(max_objects):
        progressed = False
        for source_idx, rows in enumerate(sources):
            if len(selected) >= int(max_objects):
                break
            row_idx = indexes[source_idx]
            if row_idx >= len(rows):
                continue
            selected.append(rows[row_idx])
            indexes[source_idx] += 1
            progressed = True
        if not progressed:
            break
    return selected


def _discover_assets(args: argparse.Namespace, cfg: DictConfig) -> List[Dict[str, Any]]:
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    max_objects = args.max_objects
    if max_objects is None and cfg.assets.get("max_objects") is not None:
        max_objects = int(cfg.assets.max_objects)
    allowed = tuple(str(ext).lower() for ext in cfg.assets.allowed_extensions)

    cli_sources = _cli_sources(args)
    if cli_sources:
        source_rows = []
        skipped_sources: List[str] = []
        for source in cli_sources:
            try:
                rows = _discover_from_source(
                    source,
                    project_root=project_root,
                    allowed_extensions=allowed,
                    max_objects=_source_max_objects(None, source),
                )
            except FileNotFoundError as exc:
                if not bool(source.get("skip_missing", False)):
                    raise
                name = str(source.get("name", "unknown"))
                skipped_sources.append(f"{name} ({exc})")
                print(f"Skipping asset source {name}: {exc}", file=sys.stderr)
                continue
            if rows:
                source_rows.append(rows)
        rows = _interleave_source_rows(source_rows, max_objects=max_objects)
        if not rows:
            source_name = ", ".join(str(source.get("name")) for source in cli_sources)
            skipped = (
                f" Skipped missing sources: {', '.join(skipped_sources)}."
                if skipped_sources
                else ""
            )
            raise RuntimeError(
                f"No assets discovered from {source_name}. Check that the input path "
                "exists and contains supported mesh files." + skipped
            )
        return rows

    source_rows: List[List[Dict[str, Any]]] = []
    skipped_sources: List[str] = []
    for source in cfg.assets.sources:
        if not bool(source.get("enabled", False)):
            continue
        source_dict = OmegaConf.to_container(source, resolve=True)
        assert isinstance(source_dict, dict)
        try:
            rows = _discover_from_source(
                source_dict,
                project_root=project_root,
                allowed_extensions=allowed,
                max_objects=_source_max_objects(None, source),
            )
        except FileNotFoundError as exc:
            if not bool(source.get("skip_missing", False)):
                raise
            name = str(source.get("name", "unknown"))
            skipped_sources.append(f"{name} ({exc})")
            print(f"Skipping asset source {name}: {exc}", file=sys.stderr)
            continue
        if rows:
            source_rows.append(rows)
    rows = _interleave_source_rows(source_rows, max_objects=max_objects)
    if not rows:
        skipped = (
            f" Skipped missing sources: {', '.join(skipped_sources)}."
            if skipped_sources
            else ""
        )
        raise RuntimeError(
            "No assets discovered. Pass --shapenet-hf-local-dir, "
            "--objaverse-root, --input-manifest, --asset-root, or enable "
            "ShapeNetCore/Objaverse sources in the config." + skipped
        )
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
    if not bool(cfg.assets.get("split_from_manifest", False)):
        discovered = assign_object_disjoint_splits(
            discovered,
            fractions=OmegaConf.to_container(cfg.splits.fractions, resolve=True),
            labels=[str(label) for label in cfg.splits.labels],
            seed=int(cfg.splits.seed),
        )
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

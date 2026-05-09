#!/usr/bin/env python3
"""
Build `mesh_manifest.json` for the Blender pipeline from a ModelNet40 tree.

ModelNet meshes are usually `.off`. Supported layouts:

  1) Flat (common unpacked archive):
       MODELNET_ROOT/<category>/*.off

  2) Train / test subfolders:
       MODELNET_ROOT/train/<category>/*.off
       MODELNET_ROOT/test/<category>/*.off

Optional filtering with official split lists (text files, one model id per line), e.g.:
       airplane/airplane_0627

Examples:

  python scripts/prepare_modelnet_manifest.py \\
    --modelnet-root data/modelnet40 \\
    --output data/metadata/modelnet40_manifest.json

  python scripts/prepare_modelnet_manifest.py \\
    --modelnet-root data/modelnet40 \\
    --split-file data/raw/modelnet40_train.txt \\
    --output data/metadata/modelnet_manifest_train.json

Then point `configs/render_config.yaml` paths.mesh_manifest to the generated JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _normalize_split_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Allow paths like "airplane/airplane_0627" or "airplane/airplane_0627.off"
    line = line.replace("\\", "/").strip()
    if line.endswith(".off"):
        line = line[:-4]
    return line


def load_split_file(path: Path) -> Tuple[Set[str], bool]:
    """
    Returns (allowed_keys, stem_only).

    Lines look like 'category/modelstem' or, if no slash appears in any line,
    bare stems like 'airplane_0627' (matched against path stem).
    """
    raw_lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            key = _normalize_split_line(line)
            if key:
                raw_lines.append(key)

    if not raw_lines:
        return set(), False

    has_slash = any("/" in k for k in raw_lines)
    allowed: Set[str] = set(raw_lines)
    return allowed, not has_slash


def _collect_category_off_dirs(root: Path) -> List[Tuple[Path, str]]:
    """
    Returns list of (directory_containing_off_files, category_name).

    If root/train exists, uses train/*/ and test/*/.
    Otherwise uses root/*/ assuming each subdir is a category.
    """
    train = root / "train"
    test = root / "test"
    pairs: List[Tuple[Path, str]] = []

    if train.is_dir() and test.is_dir():
        for split_dir in (train, test):
            for cat_dir in sorted(split_dir.iterdir()):
                if cat_dir.is_dir():
                    pairs.append((cat_dir, cat_dir.name))
        return pairs

    for cat_dir in sorted(root.iterdir()):
        if cat_dir.is_dir() and not cat_dir.name.startswith("."):
            pairs.append((cat_dir, cat_dir.name))
    return pairs


def iter_modelnet_off_files(root: Path) -> Iterable[Tuple[Path, str]]:
    """Yield (path_to_off, category)."""
    for folder, category in _collect_category_off_dirs(root):
        for off_path in sorted(folder.glob("*.off")):
            yield off_path, category


def split_key_for_off(off_path: Path, category: str) -> str:
    """Key comparable to split-file lines: category/modelstem."""
    stem = off_path.stem
    return f"{category}/{stem}"


def build_manifest(
    root: Path,
    *,
    split_allowed: Optional[Set[str]],
    split_stem_only: bool,
    max_total: Optional[int],
    max_per_category: Optional[int],
    seed: int,
) -> List[Dict[str, str]]:
    root = root.resolve()

    by_cat: Dict[str, List[Path]] = {}
    for off_path, category in iter_modelnet_off_files(root):
        if split_allowed is not None:
            if split_stem_only:
                if off_path.stem not in split_allowed:
                    continue
            else:
                key = split_key_for_off(off_path, category)
                if key not in split_allowed:
                    continue
        by_cat.setdefault(category, []).append(off_path)

    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    for cat in sorted(by_cat.keys()):
        paths = list(by_cat[cat])
        rng.shuffle(paths)
        if max_per_category is not None:
            paths = paths[: max_per_category]
        for p in paths:
            stem = p.stem
            rows.append(
                {
                    "object_id": f"{cat}_{stem}",
                    "category": cat,
                    "mesh_path": str(p.resolve()),
                    "dataset": "modelnet40",
                }
            )

    rng.shuffle(rows)
    if max_total is not None:
        rows = rows[: max_total]
    rows.sort(key=lambda r: (r["category"], r["object_id"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mesh_manifest.json from ModelNet40 .off files.")
    parser.add_argument(
        "--modelnet-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "modelnet40",
        help="Path to ModelNet40 root (train/ and test/ with category subfolders).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/modelnet40_manifest.json"),
        help="Output JSON path (list of {object_id, category, mesh_path, dataset}).",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional ModelNet split list (e.g. modelnet40_train.txt) — one 'category/model' per line.",
    )
    parser.add_argument("--max-total", type=int, default=None, help="Cap total models after filtering.")
    parser.add_argument("--max-per-category", type=int, default=None, help="Random subsample per category.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_allowed: Optional[Set[str]] = None
    split_stem_only = False
    if args.split_file:
        split_allowed, split_stem_only = load_split_file(args.split_file)
        if not split_allowed:
            raise SystemExit(f"No entries parsed from split file: {args.split_file}")

    manifest = build_manifest(
        args.modelnet_root,
        split_allowed=split_allowed,
        split_stem_only=split_stem_only,
        max_total=args.max_total,
        max_per_category=args.max_per_category,
        seed=args.seed,
    )
    if not manifest:
        raise SystemExit(
            "No .off files matched. Check --modelnet-root layout and --split-file paths "
            "(keys must look like 'chair/chair_0899')."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(manifest)} entries to {args.output.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Reorder Princeton ModelNet40 from the official ZIP layout to the layout this repo expects.

Official (after unzip)::

    <root>/ModelNet40/<category>/train/*.off
    <root>/ModelNet40/<category>/test/*.off

Expected by ``src/datasets/modelnet40_dataset.py``::

    <root>/train/<category>/*.off
    <root>/test/<category>/*.off

Also handles a partially-flattened tree::

    <root>/<category>/train/*.off

Usage::

    python scripts/normalize_modelnet40_layout.py
    python scripts/normalize_modelnet40_layout.py --root data/modelnet40
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_modelnet40_root(root: Path) -> bool:
    """
    Move meshes into train/<cat>/ and test/<cat>/. Returns True if anything changed.
    """
    root = root.resolve()
    changed = False

    # Pattern A: ModelNet40/<category>/{train,test}/
    bundled = root / "ModelNet40"
    if bundled.is_dir():
        sample_cats = [p for p in bundled.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if sample_cats and (sample_cats[0] / "train").is_dir():
            for cat_dir in bundled.iterdir():
                if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                    continue
                category = cat_dir.name
                for split in ("train", "test"):
                    src = cat_dir / split
                    if not src.is_dir():
                        continue
                    dst = root / split / category
                    dst.mkdir(parents=True, exist_ok=True)
                    for off in sorted(src.glob("*.off")):
                        shutil.move(str(off), str(dst / off.name))
                    changed = True
            shutil.rmtree(bundled)
            print(f"Normalized (ModelNet40/ layout) -> {root}/train|test/<category>/")
            return True

    # Pattern B: <category>/{train,test}/ at root (no top-level train/)
    if not (root / "train").is_dir():
        cat_dirs = [
            p
            for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in {"train", "test"}
        ]
        if cat_dirs and all((p / "train").is_dir() or (p / "test").is_dir() for p in cat_dirs):
            for cat_dir in cat_dirs:
                category = cat_dir.name
                for split in ("train", "test"):
                    src = cat_dir / split
                    if not src.is_dir():
                        continue
                    dst = root / split / category
                    dst.mkdir(parents=True, exist_ok=True)
                    for off in sorted(src.glob("*.off")):
                        shutil.move(str(off), str(dst / off.name))
                    changed = True
                # remove empty category folder
                try:
                    shutil.rmtree(cat_dir)
                except OSError:
                    pass
            print(f"Normalized (category/*/train layout) -> {root}/train|test/<category>/")
            return bool(changed)

    if (root / "train").is_dir() and any((root / "train").iterdir()):
        print(f"Already normalized: {root / 'train'} has categories.")
        return False

    print(
        "No recognized ModelNet40 layout found. Expected either:\n"
        f"  {root}/ModelNet40/<category>/train/*.off\n"
        f"  or {root}/<category>/train/*.off",
        file=sys.stderr,
    )
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Normalize ModelNet40 folder layout")
    p.add_argument("--root", type=Path, default=_project_root() / "data" / "modelnet40")
    args = p.parse_args()
    normalize_modelnet40_root(args.root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create an Experiment 1 render contact sheet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.metadata.manifest import load_manifest
from exp1.rendering.qc import make_contact_sheet


def _resolve_required(project_root: Path, path: str) -> Path:
    resolved = resolve_path(project_root, path)
    assert resolved is not None
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="QC or render manifest. Defaults to paths.render_qc_manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Contact sheet path. Defaults to paths.contact_sheet.",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-size", type=int, nargs=2, default=(160, 160))
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include rows that have qc_pass=false.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    input_path = (
        args.input
        if args.input is not None
        else _resolve_required(project_root, str(cfg.paths.render_qc_manifest))
    )
    output_path = (
        args.output
        if args.output is not None
        else _resolve_required(project_root, str(cfg.paths.contact_sheet))
    )
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not output_path.is_absolute():
        output_path = project_root / output_path

    df = load_manifest(input_path, validate=False)
    if "qc_pass" in df.columns and not args.include_failed:
        df = df[df["qc_pass"].astype(bool)]
    rows = df.to_dict(orient="records")

    path = make_contact_sheet(
        rows,
        output_path,
        project_root=project_root,
        columns=args.columns,
        tile_size=tuple(args.tile_size),
        max_images=args.max_images,
    )
    print(f"Wrote contact sheet: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sanity-check ShapeNetCore/Objaverse preprocessing and GLB asset outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.assets.preprocessing_sanity import (  # noqa: E402
    parse_texture_conditions,
    run_preprocessing_sanity_check,
)


def _optional_path(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    return value.expanduser()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapenetcore-root",
        type=Path,
        default=None,
        help=(
            "Local ShapeNetCore / Hugging Face ShapeNet snapshot root. This script "
            "does not download datasets."
        ),
    )
    parser.add_argument(
        "--objaverse-root",
        type=Path,
        default=None,
        help="Local Objaverse mesh directory. This script does not download datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sanity_check_preprocessing",
        help="Directory for GLB variants and metadata reports.",
    )
    parser.add_argument(
        "--num-objects-per-dataset",
        type=int,
        default=4,
        help="Maximum number of objects to sample from each dataset.",
    )
    parser.add_argument(
        "--texture-conditions",
        nargs="+",
        default=list(("photorealistic", "flat", "random_noise")),
        help=(
            "Texture conditions to export. Accepts space-separated or comma-separated "
            "values from: photorealistic, flat, random_noise."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when required datasets/outputs/checks fail.",
    )
    parser.add_argument(
        "--shapenetcore-category",
        dest="shapenetcore_categories",
        action="append",
        default=None,
        help=(
            "Optional ShapeNetCore category name or synset id filter. Repeat for "
            "multiple categories, especially when extracting ZIP snapshots."
        ),
    )
    parser.add_argument(
        "--shapenetcore-extract-archives",
        action="store_true",
        help=(
            "Extract local ShapeNetCore category ZIP archives under the output "
            "metadata directory before discovery. No downloads are performed."
        ),
    )
    parser.add_argument(
        "--target-extent",
        type=float,
        default=1.0,
        help="Maximum normalized mesh extent for generated GLBs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        texture_conditions = parse_texture_conditions(args.texture_conditions)
        report = run_preprocessing_sanity_check(
            shapenetcore_root=_optional_path(args.shapenetcore_root),
            objaverse_root=_optional_path(args.objaverse_root),
            output_dir=args.output_dir,
            num_objects_per_dataset=int(args.num_objects_per_dataset),
            texture_conditions=texture_conditions,
            seed=int(args.seed),
            strict=bool(args.strict),
            shapenetcore_categories=args.shapenetcore_categories,
            shapenetcore_extract_archives=bool(args.shapenetcore_extract_archives),
            target_extent=float(args.target_extent),
        )
    except Exception as exc:
        print(f"Preprocessing sanity check failed before report generation: {exc}")
        return 1

    summary = report["summary"]
    print("Preprocessing sanity check complete.")
    print(f"GLB variants: {summary['glb_output_dir']}")
    print(f"JSON report: {summary['json_report_path']}")
    print(f"Markdown report: {summary['markdown_report_path']}")
    print(
        "Objects: "
        f"ShapeNetCore={summary['num_shapenetcore_objects_processed']}, "
        f"Objaverse={summary['num_objaverse_objects_processed']}"
    )
    print(
        "Artifacts: "
        f"{summary['num_glb_files_written']} GLBs, "
        f"{summary['num_texture_condition_variants_written']} texture variants"
    )
    print(
        f"Checks: {summary['num_failures']} failures, "
        f"{summary['num_warnings']} warnings"
    )

    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"][:10]:
            print(f"  - {warning}")
        if len(report["warnings"]) > 10:
            print(f"  - ... {len(report['warnings']) - 10} more")

    if report["failures"]:
        print("Failures:")
        for failure in report["failures"][:20]:
            print(f"  - {failure}")
        if len(report["failures"]) > 20:
            print(f"  - ... {len(report['failures']) - 20} more")
        return 1 if args.strict else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

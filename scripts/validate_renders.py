#!/usr/bin/env python3
"""Validate Experiment 1 render outputs and write QC manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.metadata.manifest import load_manifest, save_manifest
from exp1.rendering.qc import RenderQCConfig, validate_render_rows, valid_render_rows


def _path_from_cfg(
    project_root: Path,
    explicit: Optional[Path],
    configured: str,
) -> Path:
    path = str(explicit) if explicit is not None else configured
    resolved = resolve_path(project_root, path)
    assert resolved is not None
    return resolved


def _qc_config(render_qc: Mapping[str, Any]) -> RenderQCConfig:
    return RenderQCConfig(
        min_foreground_fraction=float(render_qc.get("min_foreground_fraction", 0.01)),
        max_foreground_fraction=float(render_qc.get("max_foreground_fraction", 0.95)),
        min_depth_valid_fraction=float(render_qc.get("min_depth_valid_fraction", 0.9)),
        normal_norm_tolerance=float(render_qc.get("normal_norm_tolerance", 0.1)),
        depth_atol=float(render_qc.get("depth_atol", 1e-4)),
        normal_atol=float(render_qc.get("normal_atol", 1e-4)),
        min_random_noise_rgb_std=float(
            render_qc.get("min_random_noise_rgb_std", 0.02)
        ),
        fail_low_random_noise_variation=bool(
            render_qc.get("fail_low_random_noise_variation", False)
        ),
        fail_random_noise_image_texture=bool(
            render_qc.get("fail_random_noise_image_texture", False)
        ),
        fail_material_override_mismatch=bool(
            render_qc.get("fail_material_override_mismatch", False)
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Render manifest/status rows. Defaults to paths.render_plan_jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="QC manifest path. Defaults to paths.render_qc_manifest.",
    )
    parser.add_argument(
        "--valid-output",
        type=Path,
        default=None,
        help="Passed-render manifest path. Defaults to paths.valid_render_manifest.",
    )
    parser.add_argument(
        "--allow-failed-renders",
        action="store_true",
        help="Run file checks even when render_status is not success.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()

    input_path = _path_from_cfg(
        project_root,
        args.input,
        str(cfg.paths.render_plan_jsonl),
    )
    output_path = _path_from_cfg(
        project_root,
        args.output,
        str(cfg.paths.render_qc_manifest),
    )
    valid_output = _path_from_cfg(
        project_root,
        args.valid_output,
        str(cfg.paths.valid_render_manifest),
    )

    rows = load_manifest(input_path, validate=False).to_dict(orient="records")
    qc_df = validate_render_rows(
        rows,
        project_root=project_root,
        config=_qc_config(cfg.render.qc),
        require_render_success=not bool(args.allow_failed_renders),
    )
    valid_df = valid_render_rows(qc_df.to_dict(orient="records"))

    save_manifest(qc_df, output_path, validate=False)
    save_manifest(valid_df, valid_output, validate=False)

    print(f"Wrote QC manifest: {output_path}")
    print(f"Wrote valid render manifest: {valid_output}")
    print(f"QC passed: {int(qc_df['qc_pass'].sum())}/{len(qc_df)}")


if __name__ == "__main__":
    main()

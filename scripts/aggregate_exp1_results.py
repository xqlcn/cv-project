#!/usr/bin/env python3
"""Aggregate Experiment 1 probe metrics into long result tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path
from exp1.evaluation.comparisons import compute_texture_dependence_drops
from exp1.evaluation.metrics import (
    aggregate_prediction_bootstrap_cis,
    aggregate_probe_metrics,
    save_results_table,
)


def _resolve_required(project_root: Path, path: str) -> Path:
    resolved = resolve_path(project_root, path)
    assert resolved is not None
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument("--probe-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--texture-drops-output", type=Path, default=None)
    parser.add_argument("--bootstrap-output", type=Path, default=None)
    parser.add_argument("--baseline-texture", default="photorealistic")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty tables instead of failing when no metrics exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    results_dir = _resolve_required(project_root, str(cfg.paths.results_dir))
    probe_root = (
        args.probe_root
        if args.probe_root is not None
        else _resolve_required(project_root, str(cfg.paths.probe_output_dir))
    )
    if not probe_root.is_absolute():
        probe_root = project_root / probe_root

    output = args.output or (results_dir / "exp1_results_long.csv")
    drops_output = args.texture_drops_output or (results_dir / "exp1_texture_drops.csv")
    bootstrap_output = args.bootstrap_output or (results_dir / "exp1_bootstrap_ci.csv")
    if not output.is_absolute():
        output = project_root / output
    if not drops_output.is_absolute():
        drops_output = project_root / drops_output
    if not bootstrap_output.is_absolute():
        bootstrap_output = project_root / bootstrap_output

    results = aggregate_probe_metrics(probe_root)
    if results.empty and not args.allow_empty:
        raise RuntimeError(f"No probe metrics found under {probe_root}")
    drops = compute_texture_dependence_drops(
        results,
        baseline_texture=str(args.baseline_texture),
    )
    save_results_table(results, output)
    save_results_table(drops, drops_output)
    bootstrap_cfg = cfg.probe.evaluation.get("bootstrap", {})
    bootstrap = aggregate_prediction_bootstrap_cis(
        probe_root,
        unit_column=str(bootstrap_cfg.get("unit", "object_id")),
        n_resamples=int(bootstrap_cfg.get("n_resamples", 1000)),
        seed=int(bootstrap_cfg.get("seed", cfg.experiment.seed)),
    )
    save_results_table(bootstrap, bootstrap_output)

    print(f"Wrote long results table: {output} ({len(results)} rows)")
    print(f"Wrote texture drops table: {drops_output} ({len(drops)} rows)")
    print(f"Wrote bootstrap CI table: {bootstrap_output} ({len(bootstrap)} rows)")


if __name__ == "__main__":
    main()

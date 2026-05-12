#!/usr/bin/env python3
"""Run or prepare reproducible Experiment 1 pipeline stages."""

from __future__ import annotations

import argparse
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import (
    default_exp1_config_path,
    ensure_local_hf_home,
    load_exp1_config,
    resolve_path,
)
from exp1.features.storage import feature_cache_path
from exp1.metadata.manifest import load_manifest
from src.utils.io import read_jsonl, write_jsonl

STAGE_ALIASES = {
    "smoke_prepare": ["setup_synthetic", "render_plan", "render_chunks"],
    "prepare": ["preprocess_assets", "render_plan", "render_chunks"],
    "post_render": [
        "combine_render_status",
        "validate_renders",
        "contact_sheet",
        "labels",
    ],
    "ml": ["features", "probes", "aggregate", "figures"],
    "all": [
        "preprocess_assets",
        "render_plan",
        "render_chunks",
        "render",
        "combine_render_status",
        "validate_renders",
        "contact_sheet",
        "labels",
        "features",
        "probes",
        "aggregate",
        "figures",
    ],
}
KNOWN_STAGES = {
    "setup_synthetic",
    "preprocess_assets",
    "render_plan",
    "render_chunks",
    "render",
    "combine_render_status",
    "validate_renders",
    "contact_sheet",
    "labels",
    "features",
    "probes",
    "aggregate",
    "figures",
}


def _resolve_required(project_root: Path, path: object) -> Path:
    resolved = resolve_path(project_root, str(path))
    assert resolved is not None
    return resolved


def expand_stages(stages: Sequence[str]) -> list[str]:
    """Expand stage aliases while preserving order and removing duplicates."""
    expanded: list[str] = []
    for stage in stages:
        stage_text = str(stage)
        values = STAGE_ALIASES.get(stage_text, [stage_text])
        for value in values:
            if value not in KNOWN_STAGES:
                raise ValueError(f"Unknown Experiment 1 pipeline stage: {value}")
            if value not in expanded:
                expanded.append(value)
    return expanded


def outputs_satisfied(paths: Iterable[Path]) -> bool:
    """Return true when every output path exists."""
    out = list(paths)
    return bool(out) and all(path.exists() for path in out)


def asset_manifest_matches_enabled_sources(path: Path, cfg) -> bool:
    """Return false for stale asset manifests from disabled sources."""
    if not path.is_file():
        return False
    try:
        rows = load_manifest(path, validate=False)
    except Exception:
        return False
    if rows.empty or "source_dataset" not in rows.columns:
        return False

    disabled_datasets = set()
    for source in cfg.assets.sources:
        if bool(source.get("enabled", False)):
            continue
        dataset = source.get("source_dataset", source.get("name"))
        if dataset is not None:
            disabled_datasets.add(str(dataset))
    manifest_datasets = {str(value) for value in rows["source_dataset"].dropna()}
    return not bool(manifest_datasets & disabled_datasets)


def _run_command(
    name: str,
    command: Sequence[str],
    *,
    outputs: Sequence[Path] = (),
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if outputs_satisfied(outputs) and not force:
        print(f"[skip] {name}: outputs already exist")
        return
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"[run] {name}: {printable}")
    if dry_run:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not env.get("PYTHONPATH")
        else str(PROJECT_ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    subprocess.run(list(command), check=True, cwd=PROJECT_ROOT, env=env)


def _chunk_path(chunks_dir: Path, index: int) -> Path:
    return chunks_dir / f"chunk_{index:04d}.jsonl"


def write_render_chunks(
    render_plan: Path,
    chunks_dir: Path,
    *,
    chunk_size: int,
    shell_script: Path,
    config_path: Path,
    project_root: Path,
) -> list[Path]:
    """Split a render plan into JSONL chunks and write a Blender shell script."""
    rows = load_manifest(render_plan, validate=False).to_dict(orient="records")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    for start in range(0, len(rows), int(chunk_size)):
        path = _chunk_path(chunks_dir, len(chunk_paths))
        write_jsonl(path, rows[start : start + int(chunk_size)])
        chunk_paths.append(path)

    shell_script.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(project_root))}",
        'BLENDER_BIN="${BLENDER_BIN:-blender}"',
    ]
    for chunk in chunk_paths:
        status_path = chunk.with_suffix(".render_status.jsonl")
        args = [
            "--background",
            "--python",
            "scripts/render_blender.py",
            "--",
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
            "--chunk",
            str(chunk),
            "--status-output",
            str(status_path),
        ]
        quoted = " ".join(shlex.quote(arg) for arg in args)
        lines.append(f'"${{BLENDER_BIN}}" {quoted}')
    shell_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shell_script.chmod(shell_script.stat().st_mode | stat.S_IXUSR)
    return chunk_paths


def combine_render_status_chunks(chunks_dir: Path, output_path: Path) -> Path:
    """Combine Blender per-chunk status JSONL files into one manifest."""
    status_paths = sorted(chunks_dir.glob("chunk_*.render_status.jsonl"))
    if not status_paths:
        raise FileNotFoundError(f"No render status chunks found under {chunks_dir}")
    rows = []
    for path in status_paths:
        rows.extend(read_jsonl(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    return output_path


def expected_probe_metric_paths(cfg, project_root: Path) -> list[Path]:
    """Return configured probe metrics paths for idempotency checks."""
    root = _resolve_required(project_root, cfg.paths.probe_output_dir)
    paths = []
    for model in cfg.models.enabled:
        for layer in cfg.models.layers:
            for task in cfg.tasks.enabled:
                task_cfg = cfg.tasks.definitions[str(task)]
                if task_cfg.get("label_path") is None:
                    continue
                paths.append(
                    root / str(model) / str(layer) / str(task) / "metrics.json"
                )
    return paths


def figure_outputs_exist(figures_dir: Path) -> bool:
    return figures_dir.is_dir() and any(figures_dir.glob("*.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_exp1_config_path())
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["smoke_prepare"],
        help="Stage names or aliases: smoke_prepare, prepare, post_render, ml, all.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-train", type=int, default=3)
    parser.add_argument("--synthetic-val", type=int, default=1)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=None,
        help=(
            "Asset manifest for render_plan. Defaults to the synthetic catalog for "
            "smoke_prepare, otherwise paths.normalized_asset_manifest."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_exp1_config(args.config)
    ensure_local_hf_home(cfg)
    project_root = Path(str(cfg.paths.project_root)).expanduser().resolve()
    config_path = (
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    )
    stages = expand_stages(args.stages)

    synthetic_catalog = project_root / "data" / "synthetic_primitives" / "catalog.json"
    raw_asset_manifest = _resolve_required(project_root, cfg.paths.asset_manifest)
    normalized_asset_manifest = _resolve_required(
        project_root,
        cfg.paths.normalized_asset_manifest,
    )
    split_manifest = _resolve_required(project_root, cfg.paths.split_manifest)
    render_plan = _resolve_required(project_root, cfg.paths.render_plan_jsonl)
    chunks_dir = _resolve_required(project_root, cfg.paths.render_chunks_dir)
    render_script = chunks_dir / "run_blender_chunks.sh"
    render_status_manifest = chunks_dir / "render_status.jsonl"
    render_qc_manifest = _resolve_required(project_root, cfg.paths.render_qc_manifest)
    valid_render_manifest = _resolve_required(
        project_root, cfg.paths.valid_render_manifest
    )
    contact_sheet = _resolve_required(project_root, cfg.paths.contact_sheet)
    labels_dir = _resolve_required(project_root, cfg.paths.labels_dir)
    feature_dir = _resolve_required(project_root, cfg.paths.feature_dir)
    results_dir = _resolve_required(project_root, cfg.paths.results_dir)
    figures_dir = _resolve_required(project_root, cfg.paths.figures_dir)
    chunk_size = int(args.chunk_size or cfg.render.chunk_size)

    py = sys.executable
    if "setup_synthetic" in stages:
        _run_command(
            "setup_synthetic",
            [
                py,
                "scripts/setup_synthetic_primitives.py",
                "--root",
                str(synthetic_catalog.parent),
                "--n-train",
                str(args.synthetic_train),
                "--n-val",
                str(args.synthetic_val),
                "--seed",
                str(args.synthetic_seed),
            ],
            outputs=[synthetic_catalog],
            force=args.force,
            dry_run=args.dry_run,
        )

    if "preprocess_assets" in stages:
        preprocess_outputs = [
            raw_asset_manifest,
            normalized_asset_manifest,
            split_manifest,
        ]
        if (
            outputs_satisfied(preprocess_outputs)
            and asset_manifest_matches_enabled_sources(normalized_asset_manifest, cfg)
            and not args.force
        ):
            print("[skip] preprocess_assets: outputs already exist")
        else:
            _run_command(
                "preprocess_assets",
                [
                    py,
                    "scripts/preprocess_assets.py",
                    "--config",
                    str(config_path),
                ],
                force=True,
                dry_run=args.dry_run,
            )

    if "render_plan" in stages:
        if args.asset_manifest is not None:
            render_plan_assets = (
                args.asset_manifest
                if args.asset_manifest.is_absolute()
                else project_root / args.asset_manifest
            )
        elif "setup_synthetic" in stages and "preprocess_assets" not in stages:
            render_plan_assets = synthetic_catalog
        else:
            render_plan_assets = normalized_asset_manifest
        _run_command(
            "render_plan",
            [
                py,
                "scripts/create_render_plan.py",
                "--config",
                str(config_path),
                "--asset-manifest",
                str(render_plan_assets),
                "--output",
                str(render_plan),
            ],
            outputs=[render_plan],
            force=args.force,
            dry_run=args.dry_run,
        )

    if "render_chunks" in stages:
        if outputs_satisfied([render_script]) and not args.force:
            print("[skip] render_chunks: outputs already exist")
        else:
            print(f"[run] render_chunks: split {render_plan} into {chunks_dir}")
            if not args.dry_run:
                paths = write_render_chunks(
                    render_plan,
                    chunks_dir,
                    chunk_size=chunk_size,
                    shell_script=render_script,
                    config_path=config_path,
                    project_root=project_root,
                )
                print(f"Wrote {len(paths)} render chunk(s) and {render_script}")

    if "render" in stages:
        status_outputs = sorted(chunks_dir.glob("chunk_*.render_status.jsonl"))
        chunk_outputs = sorted(chunks_dir.glob("chunk_*.jsonl"))
        if (
            chunk_outputs
            and len(status_outputs) == len(chunk_outputs)
            and not args.force
        ):
            print("[skip] render: per-chunk status outputs already exist")
        else:
            _run_command(
                "render",
                ["bash", str(render_script)],
                force=args.force,
                dry_run=args.dry_run,
            )

    if "combine_render_status" in stages:
        if outputs_satisfied([render_status_manifest]) and not args.force:
            print("[skip] combine_render_status: outputs already exist")
        else:
            print(
                f"[run] combine_render_status: {chunks_dir} -> {render_status_manifest}"
            )
            if not args.dry_run:
                combine_render_status_chunks(chunks_dir, render_status_manifest)

    qc_input = (
        render_status_manifest if render_status_manifest.is_file() else render_plan
    )
    qc_args = [] if render_status_manifest.is_file() else ["--allow-failed-renders"]
    if "validate_renders" in stages:
        _run_command(
            "validate_renders",
            [
                py,
                "scripts/validate_renders.py",
                "--config",
                str(config_path),
                "--input",
                str(qc_input),
                "--output",
                str(render_qc_manifest),
                "--valid-output",
                str(valid_render_manifest),
                *qc_args,
            ],
            outputs=[render_qc_manifest, valid_render_manifest],
            force=args.force,
            dry_run=args.dry_run,
        )

    if "contact_sheet" in stages:
        _run_command(
            "contact_sheet",
            [
                py,
                "scripts/make_contact_sheet.py",
                "--config",
                str(config_path),
                "--input",
                str(render_qc_manifest),
                "--output",
                str(contact_sheet),
            ],
            outputs=[contact_sheet],
            force=args.force,
            dry_run=args.dry_run,
        )

    label_outputs = [
        labels_dir / f"labels_{task}.parquet"
        for task in cfg.tasks.enabled
        if cfg.tasks.definitions[str(task)].get("label_path") is not None
    ]
    if "labels" in stages:
        _run_command(
            "labels",
            [
                py,
                "scripts/build_labels.py",
                "--config",
                str(config_path),
                "--render-manifest",
                str(valid_render_manifest),
                "--labels-dir",
                str(labels_dir),
            ],
            outputs=label_outputs,
            force=args.force,
            dry_run=args.dry_run,
        )

    feature_outputs = [
        feature_cache_path(feature_dir, model_name=str(model), layer_name=str(layer))
        for model in cfg.models.enabled
        for layer in cfg.models.layers
    ]
    if "features" in stages:
        _run_command(
            "features",
            [
                py,
                "scripts/extract_exp1_features.py",
                "--config",
                str(config_path),
                "--render-manifest",
                str(valid_render_manifest),
                "--feature-dir",
                str(feature_dir),
            ],
            outputs=feature_outputs,
            force=args.force,
            dry_run=args.dry_run,
        )

    if "probes" in stages:
        _run_command(
            "probes",
            [py, "scripts/train_all_exp1_probes.py", "--config", str(config_path)],
            outputs=expected_probe_metric_paths(cfg, project_root),
            force=args.force,
            dry_run=args.dry_run,
        )

    results_output = results_dir / "exp1_results_long.csv"
    drops_output = results_dir / "exp1_texture_drops.csv"
    if "aggregate" in stages:
        _run_command(
            "aggregate",
            [
                py,
                "scripts/aggregate_exp1_results.py",
                "--config",
                str(config_path),
                "--output",
                str(results_output),
                "--texture-drops-output",
                str(drops_output),
            ],
            outputs=[results_output, drops_output],
            force=args.force,
            dry_run=args.dry_run,
        )

    if "figures" in stages:
        if figure_outputs_exist(figures_dir) and not args.force:
            print("[skip] figures: figure outputs already exist")
        else:
            _run_command(
                "figures",
                [
                    py,
                    "scripts/make_exp1_figures.py",
                    "--config",
                    str(config_path),
                    "--results",
                    str(results_output),
                    "--texture-drops",
                    str(drops_output),
                    "--output-dir",
                    str(figures_dir),
                ],
                force=args.force,
                dry_run=args.dry_run,
            )

    if "render_chunks" in stages:
        print(
            "Blender chunks are ready. Render with: "
            f"bash {shlex.quote(str(render_script))}"
        )


if __name__ == "__main__":
    main()

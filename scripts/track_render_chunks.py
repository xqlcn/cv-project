#!/usr/bin/env python3
"""Summarize Experiment 1 Blender render-chunk progress.

Reads each ``chunk_*.jsonl`` (excluding status sidecars) and the matching
``chunk_*.render_status.jsonl`` if present. Optionally verifies ``rgb.png``
exists on disk for rows marked successful.

Examples::

    PYTHONPATH=. python scripts/track_render_chunks.py \\
      --config configs/exp1_dense.yaml

    PYTHONPATH=. python scripts/track_render_chunks.py \\
      --chunks-dir data/exp1_main/manifests/render_chunks \\
      --project-root .

    # Refresh every 10 seconds (good while parallel Blender workers run)
    PYTHONPATH=. python scripts/track_render_chunks.py \\
      --config configs/exp1_main.yaml --watch 10 --verify-rgb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.config import default_exp1_config_path, load_exp1_config, resolve_path


def _chunk_jsonl_paths(chunks_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in chunks_dir.glob("chunk_*.jsonl")
        if ".render_status" not in p.name
    )


def _count_jsonl_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def _iter_status_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _status_counts(status_path: Path) -> tuple[int, Counter, int]:
    """Return (line_count, render_status histogram, missing_render_status)."""
    hist: Counter = Counter()
    missing = 0
    total = 0
    for row in _iter_status_rows(status_path):
        total += 1
        st = row.get("render_status")
        if st is None or str(st) == "":
            missing += 1
            hist["<no status>"] += 1
        else:
            hist[str(st)] += 1
    return total, hist, missing


def _verify_success_rgb(
    rows: Iterable[dict[str, Any]],
    project_root: Path,
) -> tuple[int, int]:
    """Return (checked, missing_files) for successful rows with rgb_path."""
    checked = 0
    missing = 0
    for row in rows:
        if str(row.get("render_status", "")).lower() != "success":
            continue
        rel = row.get("rgb_path")
        if not rel:
            continue
        path = Path(str(rel))
        if not path.is_absolute():
            path = project_root / path
        checked += 1
        if not path.is_file():
            missing += 1
    return checked, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=None,
        help="Override ``paths.render_chunks_dir`` from config.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repo root; resolves relative manifest paths (default: repo).",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="If >0, re-print the table every SECONDS until Ctrl-C.",
    )
    parser.add_argument(
        "--verify-rgb",
        action="store_true",
        help="For successful status rows, check that rgb_path exists on disk.",
    )
    return parser.parse_args()


def _print_report(
    *,
    chunks_dir: Path,
    render_root: Optional[Path],
    project_root: Path,
    verify_rgb: bool,
) -> None:
    chunk_paths = _chunk_jsonl_paths(chunks_dir)
    if not chunk_paths:
        print(f"No chunk_*.jsonl under {chunks_dir}", file=sys.stderr)
        return

    print(f"Chunks directory: {chunks_dir}")
    if render_root:
        print(f"Render root:      {render_root}")
    print()

    total_planned = 0
    total_status = 0
    grand = Counter()

    for chunk_path in chunk_paths:
        planned = _count_jsonl_lines(chunk_path)
        status_path = chunk_path.with_suffix(".render_status.jsonl")
        st_lines, hist, _ = _status_counts(status_path)

        total_planned += planned
        total_status += st_lines
        grand.update(hist)

        pending = max(0, planned - st_lines)
        pct = (100.0 * st_lines / planned) if planned else 0.0
        line = (
            f"{chunk_path.name:22}  planned={planned:5}  "
            f"status_lines={st_lines:5}  pending={pending:5}  "
            f"{pct:5.1f}%"
        )
        if hist:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(hist.items()))
            line += f"  [{detail}]"
        print(line)

        if verify_rgb and status_path.is_file():
            chk, miss = _verify_success_rgb(
                _iter_status_rows(status_path),
                project_root,
            )
            if chk:
                print(
                    f"    verify rgb: {chk} success rows checked, "
                    f"missing rgb file: {miss}"
                )

    print("-" * 72)
    print(
        f"{'ALL CHUNKS':22}  planned={total_planned:5}  "
        f"status_lines={total_status:5}  pending={max(0, total_planned - total_status):5}  "
        f"{(100.0 * total_status / total_planned) if total_planned else 0.0:5.1f}%"
    )
    if grand:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(grand.items()))
        print(f"  combined status: [{detail}]")

    combined = chunks_dir / "render_status.jsonl"
    if combined.is_file():
        n, ch, _ = _status_counts(combined)
        detail = ", ".join(f"{k}={v}" for k, v in sorted(ch.items()))
        print(f"\nrender_status.jsonl: {n} lines  [{detail}]")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    if args.chunks_dir is not None:
        chunks_dir = args.chunks_dir.expanduser()
        if not chunks_dir.is_absolute():
            chunks_dir = project_root / chunks_dir
        render_root = None
    elif args.config is not None:
        cfg = load_exp1_config(args.config)
        chunks_dir = resolve_path(
            project_root,
            str(cfg.paths.render_chunks_dir),
        )
        assert chunks_dir is not None
        render_root = resolve_path(project_root, str(cfg.paths.render_root))
        if not chunks_dir.is_dir():
            print(f"Missing chunks dir: {chunks_dir}", file=sys.stderr)
            sys.exit(2)
    else:
        cfg = load_exp1_config(default_exp1_config_path())
        chunks_dir = resolve_path(
            project_root,
            str(cfg.paths.render_chunks_dir),
        )
        assert chunks_dir is not None
        render_root = resolve_path(project_root, str(cfg.paths.render_root))

    chunks_dir = chunks_dir.resolve()

    if args.watch and args.watch > 0:
        try:
            while True:
                print("\033[2J\033[H", end="")
                _print_report(
                    chunks_dir=chunks_dir,
                    render_root=render_root,
                    project_root=project_root,
                    verify_rgb=args.verify_rgb,
                )
                print(f"\nWatching; refresh in {args.watch}s (Ctrl-C to stop).")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        _print_report(
            chunks_dir=chunks_dir,
            render_root=render_root,
            project_root=project_root,
            verify_rgb=args.verify_rgb,
        )


if __name__ == "__main__":
    main()

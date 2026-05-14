"""Locate the repo root so `import exp1` works when running scripts from Colab or nested zips."""

from __future__ import annotations

import os
from pathlib import Path

_MARKERS = ("exp1/__init__.py", "exp1/features/__init__.py")


def _has_markers(root: Path) -> bool:
    return all((root / Path(rel)).is_file() for rel in _MARKERS)


def repo_root(script_file: str | Path) -> Path:
    script_path = Path(script_file).resolve()
    env = os.environ.get("CV_PROJECT_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if _has_markers(candidate):
            return candidate
    for directory in script_path.parents:
        if _has_markers(directory):
            return directory
    return script_path.parents[1]

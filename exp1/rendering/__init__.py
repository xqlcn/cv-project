"""Rendering helpers for Experiment 1.

Keep this package import lightweight: Blender's Python may not have pandas,
torch, or the project ML environment installed. Planning helpers are imported
lazily so Blender-only modules like ``exp1.rendering.camera`` remain usable.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "build_render_plan",
    "build_render_plan_from_config",
    "load_asset_manifest",
    "normalize_asset_records",
]


def __getattr__(name: str):
    if name in __all__:
        grid = import_module("exp1.rendering.grid")
        return getattr(grid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

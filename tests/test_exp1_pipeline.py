from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from exp1.metadata.manifest import save_manifest
from exp1.config import load_exp1_config
from scripts.estimate_exp1_run import estimate_exp1_run
from scripts.run_exp1_pipeline import (
    asset_manifest_matches_enabled_sources,
    combine_render_status_chunks,
    expand_stages,
    outputs_satisfied,
    write_render_chunks,
)
from src.utils.io import write_jsonl


def test_expand_stages_expands_aliases_once() -> None:
    assert expand_stages(["smoke_prepare", "render_plan"]) == [
        "setup_synthetic",
        "render_plan",
        "render_chunks",
    ]
    assert expand_stages(["prepare"]) == [
        "preprocess_assets",
        "render_plan",
        "render_chunks",
    ]


def test_outputs_satisfied_requires_all_paths(tmp_path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("ok", encoding="utf-8")

    assert outputs_satisfied([first])
    assert not outputs_satisfied([first, second])


def test_asset_manifest_rejects_disabled_source(tmp_path) -> None:
    from omegaconf import OmegaConf

    manifest = tmp_path / "assets.jsonl"
    write_jsonl(
        manifest,
        [{"object_id": "m1", "source_dataset": "modelnet40"}],
    )
    cfg = OmegaConf.create(
        {
            "assets": {
                "sources": [
                    {
                        "name": "modelnet40",
                        "source_dataset": "modelnet40",
                        "enabled": False,
                    },
                    {
                        "name": "shapenet_hf",
                        "source_dataset": "shapenet",
                        "enabled": True,
                    },
                ]
            }
        }
    )

    assert not asset_manifest_matches_enabled_sources(manifest, cfg)


def test_write_render_chunks_and_shell_script(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {"render_id": f"r{idx}", "object_id": f"obj{idx}", "split": "train"}
            for idx in range(5)
        ]
    )
    render_plan = tmp_path / "render_plan.jsonl"
    save_manifest(rows, render_plan, validate=False)
    chunks_dir = tmp_path / "chunks"
    shell_script = chunks_dir / "run_blender_chunks.sh"

    chunks = write_render_chunks(
        render_plan,
        chunks_dir,
        chunk_size=2,
        shell_script=shell_script,
        config_path=tmp_path / "configs" / "exp1_smoke.yaml",
        project_root=tmp_path,
    )

    assert len(chunks) == 3
    assert [
        len(path.read_text(encoding="utf-8").strip().splitlines()) for path in chunks
    ] == [
        2,
        2,
        1,
    ]
    script = shell_script.read_text(encoding="utf-8")
    assert "scripts/render_blender.py" in script
    assert "chunk_0000.render_status.jsonl" in script
    assert os.access(shell_script, os.X_OK)


def test_combine_render_status_chunks(tmp_path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_jsonl(chunks_dir / "chunk_000.render_status.jsonl", [{"render_id": "r1"}])
    write_jsonl(chunks_dir / "chunk_001.render_status.jsonl", [{"render_id": "r2"}])

    output = combine_render_status_chunks(chunks_dir, tmp_path / "status.jsonl")

    assert output.read_text(encoding="utf-8").count("\n") == 2


def test_estimate_exp1_run_counts_bounded_config() -> None:
    cfg = load_exp1_config(Path("configs/exp1_bounded.yaml"))

    estimate = estimate_exp1_run(cfg)

    assert estimate["object_count"] == 50
    assert estimate["settings_per_object"] == 384
    assert estimate["render_count"] == 19200
    assert estimate["enabled_tasks"] == [
        "surface_normal_aggregate",
        "relative_depth_regions",
    ]

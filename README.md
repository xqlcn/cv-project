# Experiment 1: Textureless Geometry and Rendering Probes

This repository implements the controlled 3D-geometry probing experiment
described in `AGENTS.md` (kept locally, not version-controlled). It renders
the same ShapeNet/Objaverse objects in Blender under matched conditions
while varying texture, camera, lighting, and scale; trains frozen-feature
linear probes on CLIP and DINOv2 representations; and aggregates results
into the two canonical analysis artifacts:

- `outputs/exp1_rerender_analysis_figures/` — aggregate figures, summary
  CSVs, qualitative grids, and state-space plots that compare textures,
  models, layers, and tasks.
- `outputs/exp1_dense_probe_analysis/` — the standalone dense-probe
  report (`analysis.md`, `metrics_summary.csv`, figures, predictions).

> The repo is heavily I/O-driven: every stage reads/writes Hydra-resolved
> paths derived from a single config. Two configs drive the canonical
> pipeline: `configs/exp1_main.yaml` (global CLS probes) and
> `configs/exp1_dense.yaml` (dense patch-grid sub-study).

## Repository Layout

```
configs/                Hydra configs (see Configurations below)
docs/codex/             Engineering runbooks and design notes
exp1/                   Library code: assets, manifests, tasks, probes
scripts/                Command-line entry points (pipeline + analysis)
notebooks/              Colab notebooks for feature extraction and probes
tests/                  pytest suite for manifests, configs, metrics, etc.
blender/                Blender-only helpers (rendering, materials, etc.)
data/                   Generated renders, manifests, features (gitignored)
outputs/                Probe checkpoints, results, figures (gitignored)
```

## Configurations

| Config | Purpose |
|---|---|
| `configs/exp1_smoke.yaml` | Five-minute synthetic-asset smoke test. |
| `configs/exp1_mvp.yaml` | Small MVP run with a few real objects. |
| `configs/exp1_bounded.yaml` | Mid-size scaling step toward the full run. |
| `configs/exp1_full.yaml` | Largest-scale reference plan (research-only). |
| `configs/exp1_main.yaml` | **Canonical** main benchmark for the rerender analysis. |
| `configs/exp1_dense.yaml` | **Canonical** dense patch-depth/normal sub-study (inherits from `exp1_main`). |

`configs/exp1/{paths,render,tasks}.yaml` are shared building blocks
selected via Hydra `defaults`.

## Canonical Pipeline

The pipeline reproduces both `outputs/exp1_rerender_analysis_figures/` and
`outputs/exp1_dense_probe_analysis/`.

### 0. Environment

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

ShapeNet is gated. Authenticate with the Hugging Face CLI (so a token is
not stored in this repo):

```bash
HF_HOME=$PWD/data/hf_cache .venv/bin/python -m huggingface_hub.commands.huggingface_cli login
```

### 1. Prepare assets, splits, manifests, render plan

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_main.yaml --stages prepare

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_dense.yaml --stages prepare
```

Each produces:

- `data/exp1_main/manifests/...` (or `data/exp1_dense/...`)
- `data/<run>/normalized_assets/...`
- Blender chunk command files under `data/<run>/manifests/render_chunks/`

### 2. Render with Blender (local)

```bash
BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender \
RENDER_JOBS=4 \
bash data/exp1_main/manifests/render_chunks/run_blender_chunks.sh

BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender \
RENDER_JOBS=4 \
bash data/exp1_dense/manifests/render_chunks/run_blender_chunks.sh
```

Progress can be monitored with:

```bash
PYTHONPATH=. python scripts/track_render_chunks.py \
  --config configs/exp1_main.yaml --watch 10 --verify-rgb
```

### 3. Post-render: QC, labels, splits, contact sheets

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_main.yaml --stages post_render

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_dense.yaml --stages post_render
```

### 4. Frozen feature extraction (Colab recommended)

Open `notebooks/colab_exp1_zip_feature_extraction.ipynb` and run it
top-to-bottom. The notebook:

1. Zips the repo and uploads it (or pulls from Drive).
2. Builds Colab-relative manifest copies.
3. Calls `scripts/extract_exp1_features.py` (global/CLS features) against
   `configs/exp1_main.yaml`.
4. Calls `scripts/extract_exp1_patch_features.py` (patch grids + CLS)
   against `configs/exp1_dense.yaml`.
5. Syncs `data/exp1_main/features/` and `data/exp1_dense/features/` back
   to Drive.

The same `extract_*` scripts can be invoked locally with a CUDA-capable
machine; see the notebook cells for the exact CLI.

### 5. Probe training

Two Colab notebooks cover probe training:

- `notebooks/colab_exp1_zip_probe_training.ipynb` — main + dense in a
  single zip-based run. Calls:
  - `scripts/train_all_exp1_probes.py --config configs/exp1_main.yaml`
  - `scripts/train_all_dense_depth_probes.py --config configs/exp1_dense.yaml`
  - `scripts/train_all_dense_surface_normal_probes.py --config configs/exp1_dense.yaml`
  - `scripts/aggregate_exp1_results.py` and `scripts/make_exp1_figures.py`
    for both configs.
- `notebooks/colab_exp1_zip_probe_training_wandb.ipynb` — dense-only run
  with WandB logging (used to produce the dense report). Logs per-probe
  metrics to `<entity>/<project>` and mirrors aggregated results.

Both write to `outputs/exp1_main/` and/or `outputs/exp1_dense/`.

### 6. Aggregate and plot (local)

```bash
PYTHONPATH=. python scripts/aggregate_exp1_results.py --config configs/exp1_main.yaml
PYTHONPATH=. python scripts/aggregate_exp1_results.py --config configs/exp1_dense.yaml
PYTHONPATH=. python scripts/make_exp1_figures.py --config configs/exp1_main.yaml
PYTHONPATH=. python scripts/make_exp1_figures.py --config configs/exp1_dense.yaml
```

### 7. Final analysis artifacts

```bash
PYTHONPATH=. python scripts/analysis/plot_exp1_rerender_summary.py \
  --main-dir  outputs/exp1_main \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analysis/plot_exp1_dense_reliability.py \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analysis/plot_exp1_state_space_breakdown.py \
  --main-dir  outputs/exp1_main \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analysis/plot_exp1_qualitative_examples.py \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analyze_exp1_dense_probes.py
```

These commands produce:

- `outputs/exp1_rerender_analysis_figures/` — figures, CSV tables,
  qualitative grids.
- `outputs/exp1_dense_probe_analysis/` — `analysis.md` plus supporting
  figures and CSVs.

A convenience Make target wraps the same calls:

```bash
make exp1-rerender-analysis
```

## Make Targets

| Target | What it does |
|---|---|
| `make exp1-smoke` | Smoke pipeline prepare stage. |
| `make exp1-smoke-post` | Smoke pipeline post-render + ML stages. |
| `make exp1-bounded-plan` | Bounded-scale render plan only. |
| `make exp1-mvp-plan` | MVP-scale render plan only. |
| `make exp1-full-plan` | Full-scale render plan only. |
| `make exp1-main-prepare` | Canonical main prepare stage. |
| `make exp1-main-post` | Canonical main post-render stage. |
| `make exp1-dense-prepare` | Canonical dense prepare stage. |
| `make exp1-dense-post` | Canonical dense post-render stage. |
| `make exp1-results` | Aggregate both main and dense results. |
| `make exp1-figures` | Make per-config figures (main + dense). |
| `make exp1-rerender-analysis` | Run the five analysis scripts that build the final figures and the dense report. |

## Tests

```bash
pytest -q
```

`pytest` covers manifest schema, config compose checks (smoke / MVP /
bounded / full), label generation, probe metrics, and the rendering QC
helpers that do not require a Blender install.

## Key Source Modules

- `exp1/data/feature_dataset.py` — cached feature loaders for both global
  and patch caches.
- `exp1/tasks/` — per-task label builders (normals aggregate + dense,
  relative depth, viewpoint, lighting, scale).
- `exp1/probes/train.py` and `exp1/probes/train_dense_surface_normals.py`
  — linear probe heads and training loops.
- `exp1/evaluation/metrics.py` — bootstrap CI helpers used by all
  aggregators.
- `exp1/rendering/` — pure-Python helpers shared with Blender scripts.
- `scripts/run_exp1_pipeline.py` — top-level Hydra driver that wires the
  stages together.

## Local-Only Project Docs

`AGENTS.md`, `EXPERIMENT1_HANDOFF.md`, and `EXPERIMENT1_TASKS.md` are kept
on disk for reference but are **not** version-controlled. They are listed
in `.gitignore`.

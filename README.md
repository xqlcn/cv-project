# Controlled 3D Geometry Probes

This repository contains two active frozen-feature probing tracks:

- **Experiment 1** renders ShapeNet/Objaverse objects in Blender under
  matched conditions while varying texture, camera, lighting, and scale. It
  trains linear probes on frozen CLIP and DINOv2 features and produces the
  canonical rerender and dense-probe analysis artifacts.
- **Experiment 2B** uses ModelNet40 meshes to render controlled RGB, depth,
  and normal views with the `trimesh` + `pyrender` backend, then trains
  category and azimuth-viewpoint probes on frozen CLIP features.

The repo is heavily I/O-driven: pipeline stages read and write
Hydra-resolved paths, generated data lives under `data/`, and trained probes
plus analysis figures live under `outputs/`.

## Repository Layout

```text
configs/                Hydra configs for exp1 and ModelNet probes
docs/                   Runbooks and experiment notes
exp1/                   Experiment 1 library code: assets, manifests, tasks, probes
src/                    Shared datasets, renderers, feature extractors, probe training
scripts/                Command-line entry points for pipelines and analysis
notebooks/              Colab/exploratory notebooks
tests/                  pytest suite for exp1 manifests, configs, labels, metrics
blender/                Blender-only helpers
data/                   Generated renders, manifests, features (gitignored)
outputs/                Probe checkpoints, results, figures (gitignored)
```

## Configurations

| Config | Purpose |
|---|---|
| `configs/exp1_smoke.yaml` | Five-minute synthetic-asset smoke test. |
| `configs/exp1_mvp.yaml` | Small Experiment 1 run with a few real objects. |
| `configs/exp1_bounded.yaml` | Mid-size scaling step toward the full run. |
| `configs/exp1_full.yaml` | Largest-scale Experiment 1 reference plan. |
| `configs/exp1_main.yaml` | Canonical Experiment 1 global CLS benchmark. |
| `configs/exp1_dense.yaml` | Canonical dense patch-depth/normal sub-study. |
| `configs/modelnet_clip.yaml` | ModelNet40 category probe on frozen CLIP features. |
| `configs/modelnet_viewpoint_clip.yaml` | Experiment 2B azimuth probe on frozen CLIP features. |

`configs/exp1/{paths,render,tasks}.yaml` are shared Experiment 1 building
blocks selected via Hydra `defaults`.

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
export CV_PROJECT_ROOT="$PWD"
```

Experiment 1 uses gated ShapeNet data. Authenticate with the Hugging Face
CLI when running the ShapeNet/Objaverse pipeline:

```bash
HF_HOME=$PWD/data/hf_cache .venv/bin/python -m huggingface_hub.commands.huggingface_cli login
```

The ModelNet40 path does not require PyTorch3D. The active renderer is
`trimesh` + `pyrender` in `src/rendering/mesh_renderer.py`. On headless Linux
machines you may need an EGL or OSMesa OpenGL setup.

## Experiment 1 Pipeline

The Experiment 1 pipeline reproduces:

- `outputs/exp1_rerender_analysis_figures/` - aggregate figures, summary
  CSVs, qualitative grids, and state-space plots.
- `outputs/exp1_dense_probe_analysis/` - the dense-probe report
  (`analysis.md`, `metrics_summary.csv`, figures, predictions).

### 1. Prepare assets, splits, manifests, render plan

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_main.yaml --stages prepare

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_dense.yaml --stages prepare
```

Each run produces manifests under `data/<run>/manifests/`, normalized assets
under `data/<run>/normalized_assets/`, and Blender chunk command files under
`data/<run>/manifests/render_chunks/`.

### 2. Render with Blender

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

### 3. Post-render QC, labels, splits, contact sheets

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_main.yaml --stages post_render

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_dense.yaml --stages post_render
```

### 4. Frozen feature extraction and probe training

Open `notebooks/colab_exp1_zip_feature_extraction.ipynb` for feature
extraction. It calls `scripts/extract_exp1_features.py` for global/CLS
features and `scripts/extract_exp1_patch_features.py` for dense patch grids.

Probe training is covered by:

- `notebooks/colab_exp1_zip_probe_training.ipynb`
- `notebooks/colab_exp1_zip_probe_training_wandb.ipynb`

Both write to `outputs/exp1_main/` and/or `outputs/exp1_dense/`.

### 5. Aggregate and plot

```bash
PYTHONPATH=. python scripts/aggregate_exp1_results.py --config configs/exp1_main.yaml
PYTHONPATH=. python scripts/aggregate_exp1_results.py --config configs/exp1_dense.yaml
PYTHONPATH=. python scripts/make_exp1_figures.py --config configs/exp1_main.yaml
PYTHONPATH=. python scripts/make_exp1_figures.py --config configs/exp1_dense.yaml
```

Final publication-style artifacts:

```bash
PYTHONPATH=. python scripts/analysis/plot_exp1_rerender_summary.py \
  --main-dir outputs/exp1_main \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analysis/plot_exp1_dense_reliability.py \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analysis/plot_exp1_state_space_breakdown.py \
  --main-dir outputs/exp1_main \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analysis/plot_exp1_qualitative_examples.py \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analyze_exp1_dense_probes.py
```

The `make exp1-rerender-analysis` target wraps the same final analysis calls.

## Experiment 2B: ModelNet40 Viewpoint Probe

The ModelNet40 path creates one metadata row per rendered view, including
`render_path`, `depth_path`, `normal_path`, CLIP-friendly depth/normal
visualization paths, category labels, and camera azimuth/elevation.

### 1. Download and normalize ModelNet40

```bash
python scripts/download_modelnet40.py
```

If the automatic download is blocked, manually download `ModelNet40.zip` from
the Princeton ModelNet site and arrange it as:

```text
data/modelnet40/train/<category>/*.off
data/modelnet40/test/<category>/*.off
```

If you still have `data/modelnet40/ModelNet40/<category>/train|test/*.off`,
normalize it with:

```bash
python scripts/normalize_modelnet40_layout.py
```

### 2. Preprocess meshes and render views

```bash
python scripts/preprocess_modelnet40.py \
  --root data/modelnet40 \
  --split train \
  --n-views 16 \
  --image-size 224
```

The script defaults to `--max-meshes 100` for sanity runs. Use
`--max-meshes -1` for the full selected split.

### 3. Category probe

```bash
python -m src.training.extract_features --config-name=modelnet_clip
python -m src.training.train_modelnet_probe --config-name=modelnet_clip
```

The convenience wrapper runs preprocessing, CLIP feature extraction, and the
category probe:

```bash
bash scripts/run_modelnet_pipeline.sh --max-meshes -1
```

### 4. Azimuth viewpoint probe

If metadata was produced before azimuth/elevation and visualization paths were
added, backfill it once:

```bash
python scripts/backfill_viewpoint_metadata.py \
  --metadata data/processed/modelnet40/metadata/modelnet40_views.jsonl \
  --n-views 16 \
  --seed 42
```

Extract frozen CLIP features for RGB:

```bash
python -m src.training.extract_features --config-name=modelnet_viewpoint_clip
```

Depth and normal visualization modalities use the same config with path
overrides:

```bash
python -m src.training.extract_features --config-name=modelnet_viewpoint_clip \
  features.input_path_key=depth_vis_path \
  paths.feature_dir=data/features/modelnet40_viewpoint_clip_depth \
  paths.output_dir=outputs/probes/modelnet40_viewpoint_clip_depth

python -m src.training.extract_features --config-name=modelnet_viewpoint_clip \
  features.input_path_key=normal_vis_path \
  paths.feature_dir=data/features/modelnet40_viewpoint_clip_normal \
  paths.output_dir=outputs/probes/modelnet40_viewpoint_clip_normal
```

Train the azimuth probe. It regresses a sine/cosine target and reports angular
MAE in degrees:

```bash
python -m src.training.train_viewpoint_probe --config-name=modelnet_viewpoint_clip

python -m src.training.train_viewpoint_probe --config-name=modelnet_viewpoint_clip \
  paths.feature_dir=data/features/modelnet40_viewpoint_clip_depth \
  paths.output_dir=outputs/probes/modelnet40_viewpoint_clip_depth
```

See `docs/experiment2b_viewpoint_probe.md` for a compact runbook and artifact
map.

## Dataset API

The shared dataset exports are lazy, so `import src.datasets` stays light until
a specific renderer-backed dataset is requested.

```python
from src.datasets import (
    ModelNet40MeshDataset,
    RenderedModelNetDataset,
    RenderedSyntheticPrimitiveDataset,
)

mesh_ds = ModelNet40MeshDataset(split="train")
sample = mesh_ds[0]  # mesh_path, category, split, object_id, category_id, dataset
```

Rendered samples use a flat one-view-per-row schema when `flatten_views=True`:

```python
{
    "mesh_path": str,
    "category": str,
    "split": str,
    "view_id": int,
    "object_id": str,
    "dataset": str,
    "rgb": np.uint8,      # [H, W, 3]
    "depth": np.float32,  # [H, W]
    "normal": np.float32, # [H, W, 3]
}
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
| `make exp1-figures` | Make per-config figures for main and dense runs. |
| `make exp1-rerender-analysis` | Build the final rerender figures and dense report. |

## Tests

```bash
pytest -q
```

The existing pytest suite covers Experiment 1 manifest schema, config compose
checks, label generation, probe metrics, and rendering QC helpers that do not
require a Blender install.

## Local-Only Project Docs

`AGENTS.md`, `EXPERIMENT1_HANDOFF.md`, and `EXPERIMENT1_TASKS.md` are kept
on disk for reference but are **not** version-controlled. They are listed in
`.gitignore`.

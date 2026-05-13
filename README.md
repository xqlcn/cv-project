# Controlled 3D geometry probing (CLIP & DINOv2)

This repo supports **frozen** CLIP / DINOv2 features + **linear probes**, plus two mesh pipelines:

1. **Experiment 1 controlled pipeline:** ShapeNetCore/Objaverse meshes → Blender texture controls → **RGB**, **depth**, **normal**, and **mask** buffers.
2. **Smoke fixtures:** synthetic primitives for tiny local pipeline checks.

An optional **Blender** path (`blender/render_dataset.py`) remains for high-quality chirality renders using a JSON manifest (no ShapeNet dependency).

## Layout

- `src/datasets/` — synthetic cache and rendered `Dataset` wrappers
- `src/rendering/` — multi-view pyrender backend (RGB / depth / normals)
- `blender/` — headless Blender chirality pipeline + manifest
- `configs/` — YAML / Hydra
- `data/shapenet_hf/` — optional local ShapeNetCore snapshots/extractions
- `data/objaverse/` and `data/objaverse_cache/` — optional local Objaverse assets/cache
- `data/synthetic_primitives/` — generated `.obj` primitives + `catalog.json`

## Setup (Python)

```bash
cd "/path/to/CV final project"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# PyTorch: pick the wheel that matches your machine from https://pytorch.org
export PYTHONPATH="$PWD"
export CV_PROJECT_ROOT="$PWD"
```

**`pyrender` / `PyOpenGL`:** `requirements.txt` installs **pyrender from GitHub** (not PyPI 0.1.45) so pip can resolve **`PyOpenGL>=3.1.7`** (PyPI pyrender wrongly pins `3.1.0`, which breaks Python 3.7+). You need **`git`** on your PATH for that line.

If you cannot use `git+https`, install PyPI pyrender without pulling its OpenGL pin, then upgrade OpenGL:

```bash
pip install "pyrender==0.1.45" --no-deps
pip install "PyOpenGL>=3.1.7,<3.2"
pip install trimesh networkx scipy pyglet "Pillow>=10" imageio freetype-py six
```

**Headless rendering:** on Linux servers you may need `export PYOPENGL_PLATFORM=osmesa` (and OSMesa installed) or EGL; on macOS the default often works for offscreen pyrender.

## 1) Synthetic primitives (default experiment)

```bash
python scripts/setup_synthetic_primitives.py
```

Creates `data/synthetic_primitives/train/*.obj`, `val/*.obj`, and `catalog.json`.

## 2) ShapeNetCore from Hugging Face

After your Hugging Face account has access to the gated
`ShapeNet/ShapeNetCore` repository, log in locally and preprocess a small
category subset first:

```bash
huggingface-cli login
python scripts/preprocess_assets.py \
  --config configs/exp1_mvp.yaml \
  --shapenet-hf-repo-id ShapeNet/ShapeNetCore \
  --shapenet-hf-local-dir data/shapenet_hf/ShapeNetCore \
  --shapenet-category chair \
  --shapenet-category table \
  --max-objects 50 \
  --overwrite
```

`ShapeNet/ShapeNetCore` is stored as per-category synset ZIPs; preprocessing
downloads only the requested category ZIPs when `--shapenet-category` is set,
extracts them into `data/shapenet_hf/extracted/`, and normalizes the meshes.
This writes the normal Experiment 1 asset manifests under `data/exp1/manifests/`
and normalized GLBs under `data/exp1/normalized_assets/`, so downstream render
planning can keep using:

```bash
python scripts/create_render_plan.py \
  --config configs/exp1_mvp.yaml \
  --asset-manifest data/exp1/manifests/assets_normalized.jsonl
```

Once you have access to the GLB mirror, use the same command with
`--shapenet-hf-repo-id ShapeNet/shapenetcore-glb` and a separate local dir such
as `data/shapenet_hf/shapenetcore-glb`.
Use `--shapenet-no-download` with `--shapenet-hf-local-dir` to scan an existing
snapshot without contacting Hugging Face.

## 3) Objaverse Bounded Download

Objaverse access is optional and bounded by config defaults in
`configs/exp1/paths.yaml` (`datasets.objaverse.max_objects` and
`datasets.objaverse.max_download_gb`). The downloader writes a normal asset
manifest that `scripts/preprocess_assets.py` can consume alongside ShapeNet:

```bash
python scripts/download_objaverse_assets.py \
  --config configs/exp1_mvp.yaml \
  --category chair \
  --category table \
  --max-objects 50 \
  --max-download-gb 5
```

Then prepare a render plan from configured ShapeNet/Objaverse sources without
running Blender:

```bash
make exp1-mvp-plan
```

## 4) Bounded Depth/Normal Run

Use `configs/exp1_bounded.yaml` as the next non-toy scale-up config. It keeps
the core geometry probes only:

- `surface_normal_aggregate`;
- `relative_depth_regions`.

Before rendering, estimate the render count and storage:

```bash
make exp1-bounded-estimate
```

Prepare assets and render chunks without launching Blender:

```bash
make exp1-bounded-plan
```

Then render and finish the pipeline:

```bash
BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender" \
  bash data/exp1_bounded/manifests/render_chunks/run_blender_chunks.sh

HF_HOME=data/hf_cache PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_bounded.yaml \
  --stages post_render ml
```

## 5) Experiment 1 smoke pipeline

The safe smoke command prepares a tiny synthetic asset catalog, writes an
Experiment 1 render plan, exports JSONL render chunks, and creates a Blender
shell script. Stages are idempotent; rerunning the command skips outputs that
already exist unless `--force` is passed.

```bash
make exp1-smoke
```

Render the prepared chunks with Blender:

```bash
bash data/exp1/manifests/render_chunks/run_blender_chunks.sh
```

After Blender finishes, build QC, labels, features, probes, aggregated results,
and figures:

```bash
make exp1-smoke-post
```

Equivalent direct runner commands:

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_smoke.yaml \
  --stages smoke_prepare

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_smoke.yaml \
  --stages post_render ml
```

Standalone reporting commands:

```bash
PYTHONPATH=. python scripts/aggregate_exp1_results.py --config configs/exp1_smoke.yaml
PYTHONPATH=. python scripts/make_exp1_figures.py --config configs/exp1_smoke.yaml
```

## 6) Rendered sample schema (synthetic + ShapeNet/Objaverse corpora)

Each **rendered** training sample is a flat dict (one **view** per row when `flatten_views=True`):

```python
{
    "mesh_path": str,
    "category": str,
    "split": str,
    "view_id": int,
    "object_id": str,
    "dataset": str,  # "synthetic_primitives" | "shapenet" | "objaverse" | ...
    "rgb": np.uint8[H, W, 3],
    "depth": np.float32[H, W],   # linear depth in meters (inf = background)
    "normal": np.float32[H, W, 3],  # camera-space normals in [-1, 1]
}
```

**PyTorch datasets**

- `RenderedSyntheticPrimitiveDataset` — default controlled renders.
- `RenderedMeshDataset` — generic wrapper over any list of records with the same keys (e.g. ShapeNet/Objaverse rows with `mesh_path` + `category` + `split`).

Example:

```python
from src.rendering.mesh_renderer import RenderConfig
from src.datasets import RenderedSyntheticPrimitiveDataset

ds = RenderedSyntheticPrimitiveDataset(
    split="train",
    render_cfg=RenderConfig(image_size=(224, 224), n_views=8, seed=0),
)
sample = ds[0]  # keys: mesh_path, category, split, view_id, rgb, depth, normal, ...
```

**PyTorch3D:** not required. The active backend is **trimesh + pyrender** (`src/rendering/mesh_renderer.py`). Optional PyTorch3D hook lives in `src/rendering/pytorch3d_backend.py` (stub for you to implement if you install `pytorch3d`).

## Feature extraction & probes

Use the Experiment 1 pipeline scripts for current runs:

```bash
PYTHONPATH=. python scripts/extract_exp1_features.py --config configs/exp1_smoke.yaml
PYTHONPATH=. python scripts/train_all_exp1_probes.py --config configs/exp1_smoke.yaml
```

## Hydra

```bash
python -m src.training.train_probe --config-name=clip_probe probe.epochs=5
```

## Licenses

- **ShapeNet** / **Objaverse** — follow each dataset's terms for redistribution and citation.
- **CLIP**, **DINOv2**, **OpenCLIP** — respect each model license in publications.

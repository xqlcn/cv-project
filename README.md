# Controlled 3D geometry probing (CLIP & DINOv2)

This repo supports **frozen** CLIP / DINOv2 features + **linear probes**, plus two mesh pipelines:

1. **Default (controlled):** synthetic primitives → `trimesh` + `pyrender` multi-view **RGB**, **depth**, **normal** maps (`src/datasets/synthetic_primitives.py`).
2. **Realistic validation:** **ModelNet40** under `data/modelnet40/{train,test}/<category>/*.off` (`src/datasets/modelnet40_dataset.py`).

An optional **Blender** path (`blender/render_dataset.py`) remains for high-quality chirality renders using a JSON manifest (no ShapeNet dependency).

## Layout

- `src/datasets/` — ModelNet40 index, synthetic cache, rendered `Dataset` wrappers
- `src/rendering/` — multi-view pyrender backend (RGB / depth / normals)
- `blender/` — headless Blender chirality pipeline + manifest
- `configs/` — YAML / Hydra
- `data/modelnet40/` — expected ModelNet40 root after download
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

## 1) ModelNet40 download

Automatic (may fail if the server blocks bots; then use manual steps printed by the script):

```bash
python scripts/download_modelnet40.py
```

Manual fallback:

1. Open [https://modelnet.cs.princeton.edu/](https://modelnet.cs.princeton.edu/) and download **ModelNet40.zip**.
2. Unzip so you have `data/modelnet40/train/<category>/*.off` and `data/modelnet40/test/<category>/*.off`.

Optional Blender manifest from the same tree:

```bash
python scripts/prepare_modelnet_manifest.py \
  --modelnet-root data/modelnet40 \
  --output data/metadata/modelnet40_manifest.json
```

## 2) Synthetic primitives (default experiment)

```bash
python scripts/setup_synthetic_primitives.py
```

Creates `data/synthetic_primitives/train/*.obj`, `val/*.obj`, and `catalog.json`.

## 2b) ShapeNetCore from Hugging Face

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

## 2c) Experiment 1 smoke pipeline

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

## 3) Rendered sample schema (ModelNet + synthetic + future corpora)

Each **rendered** training sample is a flat dict (one **view** per row when `flatten_views=True`):

```python
{
    "mesh_path": str,
    "category": str,
    "split": str,
    "view_id": int,
    "object_id": str,
    "dataset": str,  # "modelnet40" | "synthetic_primitives" | ...
    "rgb": np.uint8[H, W, 3],
    "depth": np.float32[H, W],   # linear depth in meters (inf = background)
    "normal": np.float32[H, W, 3],  # camera-space normals in [-1, 1]
}
```

**PyTorch datasets**

- `RenderedSyntheticPrimitiveDataset` — default controlled renders.
- `RenderedModelNetDataset` — ModelNet40 validation / training renders.
- `RenderedMeshDataset` — generic wrapper over any list of records with the same keys (e.g. future ShapeNet rows with `mesh_path` + `category` + `split`).

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

## Blender (optional) chirality pipeline

1. Install Blender 3.6+.
2. Install **PyYAML** into Blender’s Python (see previous README sections).
3. Point `configs/render_config.yaml` → `paths.mesh_manifest` at `data/metadata/modelnet40_manifest.json`.
4. Run `./scripts/render_chirality_dataset.sh`.

## Feature extraction & probes (unchanged)

```bash
./scripts/extract_clip_features.sh
./scripts/extract_dino_features.sh
./scripts/train_clip_probe.sh
./scripts/train_dino_probe.sh
```

## Hydra

```bash
python -m src.training.train_probe --config-name=clip_probe probe.epochs=5
```

## Licenses

- **ModelNet** (Princeton) — follow their terms for redistribution and citation.
- **CLIP**, **DINOv2**, **OpenCLIP** — respect each model license in publications.

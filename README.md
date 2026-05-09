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

The Princeton ZIP unpacks as `ModelNet40/<category>/train|test/`; the download script (or the command below) flattens that to `train/<category>/` and `test/<category>/`.

If you unzipped by hand and still see a `ModelNet40` folder inside `data/modelnet40/`:

```bash
python scripts/normalize_modelnet40_layout.py
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

## ModelNet40 End-To-End (your 10 steps)

Preprocessing + probing pipeline commands:

```bash
# Step 1-8 (scan, load .off, triangulate, center, scale, fix normals, render 16 views, save metadata)
python scripts/preprocess_modelnet40.py --root data/modelnet40 --split train --n-views 16 --image-size 224

# Step 9 (frozen CLIP extraction)
python -m src.training.extract_features --config-name=modelnet_clip

# Step 10 (linear probe on category_id labels)
python -m src.training.train_modelnet_probe --config-name=modelnet_clip
```

Or run all three with:

```bash
bash scripts/run_modelnet_pipeline.sh
```

## Experiment 2B: Viewpoint probe (azimuth first)

1) If your processed metadata was generated before azimuth/elevation and vis paths were added, backfill once:

```bash
python scripts/backfill_viewpoint_metadata.py --metadata data/processed/modelnet40/metadata/modelnet40_views.jsonl --n-views 16 --seed 42
```

2) Extract frozen CLIP features for your chosen modality:

```bash
# RGB
python -m src.training.extract_features --config-name=modelnet_viewpoint_clip

# Depth visualization
python -m src.training.extract_features --config-name=modelnet_viewpoint_clip \
  features.input_path_key=depth_vis_path \
  paths.feature_dir=data/features/modelnet40_viewpoint_clip_depth \
  paths.output_dir=outputs/probes/modelnet40_viewpoint_clip_depth

# Normal visualization
python -m src.training.extract_features --config-name=modelnet_viewpoint_clip \
  features.input_path_key=normal_vis_path \
  paths.feature_dir=data/features/modelnet40_viewpoint_clip_normal \
  paths.output_dir=outputs/probes/modelnet40_viewpoint_clip_normal
```

3) Train azimuth probe (linear regression on sin/cos target; reports angular MAE in degrees):

```bash
# RGB
python -m src.training.train_viewpoint_probe --config-name=modelnet_viewpoint_clip

# Depth / normal: use matching feature_dir + output_dir overrides
python -m src.training.train_viewpoint_probe --config-name=modelnet_viewpoint_clip \
  paths.feature_dir=data/features/modelnet40_viewpoint_clip_depth \
  paths.output_dir=outputs/probes/modelnet40_viewpoint_clip_depth
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

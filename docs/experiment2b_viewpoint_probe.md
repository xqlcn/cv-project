# Experiment 2B Viewpoint Probe Runbook

Experiment 2B renders ModelNet40 meshes from controlled camera views and
trains linear probes on frozen CLIP features. The first target is azimuth:
`src/training/train_viewpoint_probe.py` regresses `[sin(azimuth), cos(azimuth)]`
and reports validation angular MAE in degrees.

## Inputs

- ModelNet40 meshes under `data/modelnet40/{train,test}/<category>/*.off`.
- Metadata rows written by `scripts/preprocess_modelnet40.py` to
  `data/processed/modelnet40/metadata/modelnet40_views.jsonl`.
- Frozen feature indexes written by `src.training.extract_features` to
  `data/features/<run>/features_index.jsonl`.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
export CV_PROJECT_ROOT="$PWD"
```

The renderer uses `trimesh` + `pyrender`; PyTorch3D is not required.

## Step 1: Download Or Normalize ModelNet40

```bash
python scripts/download_modelnet40.py
```

If the Princeton download blocks automated requests, download the ZIP manually
and arrange files as:

```text
data/modelnet40/train/<category>/*.off
data/modelnet40/test/<category>/*.off
```

If the archive is still nested as `data/modelnet40/ModelNet40/...`, run:

```bash
python scripts/normalize_modelnet40_layout.py
```

## Step 2: Preprocess And Render

```bash
python scripts/preprocess_modelnet40.py \
  --root data/modelnet40 \
  --split train \
  --n-views 16 \
  --image-size 224 \
  --max-meshes -1
```

The preprocessing script scans `.off` files, triangulates and normalizes
meshes, renders RGB/depth/normal views, saves depth and normal visualizations,
and writes per-view metadata with camera azimuth/elevation.

## Step 3: Optional Category Probe

```bash
python -m src.training.extract_features --config-name=modelnet_clip
python -m src.training.train_modelnet_probe --config-name=modelnet_clip
```

The wrapper below runs preprocessing, feature extraction, and category-probe
training. Extra arguments are forwarded to preprocessing:

```bash
bash scripts/run_modelnet_pipeline.sh --max-meshes -1
```

## Step 4: Azimuth Probe

Extract RGB features:

```bash
python -m src.training.extract_features --config-name=modelnet_viewpoint_clip
```

Extract depth visualization features:

```bash
python -m src.training.extract_features --config-name=modelnet_viewpoint_clip \
  features.input_path_key=depth_vis_path \
  paths.feature_dir=data/features/modelnet40_viewpoint_clip_depth \
  paths.output_dir=outputs/probes/modelnet40_viewpoint_clip_depth
```

Extract normal visualization features:

```bash
python -m src.training.extract_features --config-name=modelnet_viewpoint_clip \
  features.input_path_key=normal_vis_path \
  paths.feature_dir=data/features/modelnet40_viewpoint_clip_normal \
  paths.output_dir=outputs/probes/modelnet40_viewpoint_clip_normal
```

Train the RGB azimuth probe:

```bash
python -m src.training.train_viewpoint_probe --config-name=modelnet_viewpoint_clip
```

For depth or normal probes, point training at the matching feature and output
directories:

```bash
python -m src.training.train_viewpoint_probe --config-name=modelnet_viewpoint_clip \
  paths.feature_dir=data/features/modelnet40_viewpoint_clip_depth \
  paths.output_dir=outputs/probes/modelnet40_viewpoint_clip_depth
```

## Artifacts

- `data/processed/modelnet40/images/` - RGB render PNGs.
- `data/processed/modelnet40/depth/` - depth arrays and visualization PNGs.
- `data/processed/modelnet40/normal/` - normal arrays and visualization PNGs.
- `data/processed/modelnet40/meshes/` - normalized OBJ meshes.
- `data/processed/modelnet40/metadata/modelnet40_views.jsonl` - per-view rows.
- `data/features/modelnet40_viewpoint_clip_*` - frozen feature tensors and index.
- `outputs/probes/modelnet40_viewpoint_clip_*` - trained azimuth probe checkpoints.

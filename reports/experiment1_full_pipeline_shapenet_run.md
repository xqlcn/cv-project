# Experiment 1 ShapeNet Pipeline Run

## Overall Status

**Completed: reduced local ShapeNet full-pipeline run.** The pipeline ran end-to-end from local ShapeNetCore OBJ assets through normalized manifests, Blender renders, QC, labels, CLIP/DINOv2 feature extraction, linear probe training, aggregation, bootstrap tables, and figures.

This was **not the final full-scale Experiment 1 dataset run**. It used 3 local ShapeNetCore chair objects and a reduced render grid so the pipeline could complete locally on CPU. Unlike the previous reduced ModelNet run, this run used ShapeNet assets and included CLIP ViT-B/16, CLIP ViT-L/14, and DINOv2 ViT-B.

## Commands To Run The Full Pipeline

Canonical full experiment commands, using the configured ShapeNet/Objaverse sources:

```bash
export PYTHONPATH=.
export CV_PROJECT_ROOT="$PWD"
export HF_HOME="$PWD/data/hf_cache"

# Optional: populate a bounded Objaverse manifest before preprocessing.
.venv/bin/python scripts/download_objaverse_assets.py \
  --config configs/exp1_full.yaml \
  --max-objects 250 \
  --max-download-gb 20

# Prepare assets and Blender render chunks. This does not render yet.
.venv/bin/python scripts/run_exp1_pipeline.py \
  --config configs/exp1_full.yaml \
  --stages prepare

# Render the prepared chunks.
BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender" \
  bash data/exp1/manifests/render_chunks/run_blender_chunks.sh

# Build QC, contact sheet, and labels.
.venv/bin/python scripts/run_exp1_pipeline.py \
  --config configs/exp1_full.yaml \
  --stages post_render

# Extract features, train probes, aggregate results, and make figures.
HF_HOME="$PWD/data/hf_cache" \
  .venv/bin/python scripts/run_exp1_pipeline.py \
  --config configs/exp1_full.yaml \
  --stages ml
```

Practical warning: `configs/exp1_full.yaml` is a true final-grid config. With hundreds of objects it can expand to millions of render rows, so run it only on adequate storage/compute. The local cross-check run below used `reports/experiment1_full_pipeline_shapenet_run_config.yaml`.

## Environment

- OS: macOS 14.6 arm64.
- Project venv Python: `Python 3.9.6`.
- PyTorch: `2.8.0`.
- CUDA: unavailable.
- PyTorch MPS: unavailable in this venv.
- Blender: 5.1.1 at `/Applications/Blender.app/Contents/MacOS/Blender`.
- DINOv2 loader: Transformers backend with `facebook/dinov2-base`.

## Commands Run

```bash
PYTHONPATH=. .venv/bin/python scripts/preprocess_assets.py \
  --config reports/experiment1_full_pipeline_shapenet_run_config.yaml \
  --shapenet-hf-local-dir data/shapenet_hf/ShapeNetCore \
  --shapenet-no-download \
  --shapenet-category chair \
  --max-objects 3 \
  --overwrite

PYTHONPATH=. .venv/bin/python scripts/create_render_plan.py \
  --config reports/experiment1_full_pipeline_shapenet_run_config.yaml \
  --asset-manifest data/exp1_full_pipeline_shapenet_run/manifests/assets_normalized.jsonl

PYTHONPATH=. .venv/bin/python scripts/run_exp1_pipeline.py \
  --config reports/experiment1_full_pipeline_shapenet_run_config.yaml \
  --stages render_chunks \
  --force

BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender" \
  bash data/exp1_full_pipeline_shapenet_run/manifests/render_chunks/run_blender_chunks.sh

PYTHONPATH=. .venv/bin/python scripts/run_exp1_pipeline.py \
  --config reports/experiment1_full_pipeline_shapenet_run_config.yaml \
  --stages post_render \
  --force

PYTHONPATH=. .venv/bin/python scripts/run_exp1_pipeline.py \
  --config reports/experiment1_full_pipeline_shapenet_run_config.yaml \
  --stages ml \
  --force

HF_HOME=data/hf_cache PYTHONPATH=. .venv/bin/python scripts/extract_exp1_features.py \
  --config reports/experiment1_full_pipeline_shapenet_run_config.yaml \
  --render-manifest data/exp1_full_pipeline_shapenet_run/manifests/render_valid.parquet \
  --feature-dir data/exp1_full_pipeline_shapenet_run/features \
  --models dinov2_vit_b \
  --layers final layer4 layer8 layer12 \
  --device cpu

PYTHONPATH=. .venv/bin/python scripts/run_exp1_pipeline.py \
  --config reports/experiment1_full_pipeline_shapenet_run_config.yaml \
  --stages probes aggregate figures \
  --force
```

The first `ml` attempt completed CLIP feature extraction but failed when DINOv2 tried to write to the user-level Hugging Face cache. Rerunning DINOv2 with `HF_HOME=data/hf_cache` fixed it, after which probes/aggregation/figures completed.

## Dataset Availability

- `data/shapenet_hf/ShapeNetCore`: found and used.
- Local ShapeNet contents detected: 6,778 `.obj` files and 1 category ZIP under the local ShapeNetCore tree.
- `data/objaverse`: not found.
- `data/objaverse_cache`: not found before this run.
- `data/exp1/manifests/objaverse_assets.jsonl`: not found before this run.

Run subset:

- Dataset used: ShapeNetCore.
- Objects processed: 3.
- Categories processed: 1 (`chair`).
- Object-disjoint splits: 1 train object, 1 val object, 1 test object.
- Render rows planned: 72.
- Texture rows planned: 24 each for `photorealistic`, `flat`, and `random_noise`.

## Generated Artifacts

- Run config: `reports/experiment1_full_pipeline_shapenet_run_config.yaml`.
- Raw asset manifest: `data/exp1_full_pipeline_shapenet_run/manifests/assets.jsonl`.
- Normalized asset manifest: `data/exp1_full_pipeline_shapenet_run/manifests/assets_normalized.jsonl`.
- Object split manifest: `data/exp1_full_pipeline_shapenet_run/manifests/object_splits.jsonl`.
- Render plan: `data/exp1_full_pipeline_shapenet_run/manifests/render_plan.jsonl`.
- Render chunks and shell script: `data/exp1_full_pipeline_shapenet_run/manifests/render_chunks/`.
- Combined render status: `data/exp1_full_pipeline_shapenet_run/manifests/render_chunks/render_status.jsonl`.
- Renders: `data/exp1_full_pipeline_shapenet_run/renders/`.
- QC manifest: `data/exp1_full_pipeline_shapenet_run/manifests/render_qc.parquet`.
- Valid render manifest: `data/exp1_full_pipeline_shapenet_run/manifests/render_valid.parquet`.
- Contact sheet: `data/exp1_full_pipeline_shapenet_run/qc/contact_sheet.png`.
- Labels: `data/exp1_full_pipeline_shapenet_run/labels/*.parquet`.
- Feature caches: `data/exp1_full_pipeline_shapenet_run/features/{clip_vit_b16,clip_vit_l14,dinov2_vit_b}/*.npz`.
- Probe outputs: `outputs/exp1_full_pipeline_shapenet_run/probes/`.
- Results tables: `outputs/exp1_full_pipeline_shapenet_run/results/exp1_results_long.csv`, `exp1_texture_drops.csv`, `exp1_bootstrap_ci.csv`.
- Figures: `outputs/exp1_full_pipeline_shapenet_run/figures/`.

## Produced Labels

- `labels_surface_normal_aggregate.parquet`: 72 rows, 72 valid.
- `labels_relative_depth_regions.parquet`: 72 rows, 72 valid.
- `labels_camera.parquet`: 72 rows, 72 valid.
- `labels_lighting.parquet`: 72 rows, 72 valid.
- `labels_scale.parquet`: 72 rows, 72 valid.

Note: this reduced grid varies camera distance, azimuth, and lighting intensity. Camera elevation, light direction, and object scale are fixed, so those task results are pipeline checks, not scientifically meaningful estimates.

## Validation Checks

Passed:

- Asset preprocessing: 3 ShapeNet assets normalized, 0 failed.
- Render IDs: no duplicates in 72 planned rows.
- Blender rendering: 72 render attempts wrote success status rows.
- QC: 72/72 renders passed and were retained.
- Texture-control geometry QC: passed after fixing raw-vs-normalized mesh import consistency.
- Feature caches: CLIP ViT-B/16, CLIP ViT-L/14, and DINOv2 ViT-B caches are finite and contain render IDs.
- Probe training: 252 probe jobs completed.
- Results: 3,276 long-result rows, 2,112 texture-drop rows, 972 bootstrap rows, and 181 figures were produced.

Feature cache shapes:

| model | layer | shape |
|---|---|---:|
| clip_vit_b16 | final | `(72, 512)` |
| clip_vit_b16 | layer4/layer8/layer12 | `(72, 768)` |
| clip_vit_l14 | final | `(72, 768)` |
| clip_vit_l14 | layer4/layer8/layer12 | `(72, 1024)` |
| dinov2_vit_b | final/layer4/layer8/layer12 | `(72, 768)` |

## Results Summary

These are final-layer test metrics from the reduced ShapeNet run. Lower is better for angular/MAE metrics; higher is better for relative-depth accuracy.

| task | model | texture | metric | test value |
|---|---|---|---|---:|
| apparent_scale | clip_vit_b16 | flat | mae_mean | 0.3435 |
| apparent_scale | clip_vit_b16 | photorealistic | mae_mean | 0.3561 |
| apparent_scale | clip_vit_b16 | random_noise | mae_mean | 0.3541 |
| apparent_scale | clip_vit_l14 | flat | mae_mean | 0.4861 |
| apparent_scale | clip_vit_l14 | photorealistic | mae_mean | 0.4648 |
| apparent_scale | clip_vit_l14 | random_noise | mae_mean | 0.2283 |
| apparent_scale | dinov2_vit_b | flat | mae_mean | 0.6943 |
| apparent_scale | dinov2_vit_b | photorealistic | mae_mean | 0.7390 |
| apparent_scale | dinov2_vit_b | random_noise | mae_mean | 0.5500 |
| camera_distance | clip_vit_b16 | flat | mae_mean | 0.5553 |
| camera_distance | clip_vit_b16 | photorealistic | mae_mean | 0.5032 |
| camera_distance | clip_vit_b16 | random_noise | mae_mean | 0.4357 |
| camera_distance | clip_vit_l14 | flat | mae_mean | 0.4072 |
| camera_distance | clip_vit_l14 | photorealistic | mae_mean | 0.4050 |
| camera_distance | clip_vit_l14 | random_noise | mae_mean | 0.1396 |
| camera_distance | dinov2_vit_b | flat | mae_mean | 0.5081 |
| camera_distance | dinov2_vit_b | photorealistic | mae_mean | 0.9386 |
| camera_distance | dinov2_vit_b | random_noise | mae_mean | 0.4429 |
| lighting_direction | clip_vit_b16 | flat | angular_error_deg_mean | 33.7463 |
| lighting_direction | clip_vit_b16 | photorealistic | angular_error_deg_mean | 33.7050 |
| lighting_direction | clip_vit_b16 | random_noise | angular_error_deg_mean | 33.7831 |
| lighting_direction | clip_vit_l14 | flat | angular_error_deg_mean | 17.5333 |
| lighting_direction | clip_vit_l14 | photorealistic | angular_error_deg_mean | 19.3000 |
| lighting_direction | clip_vit_l14 | random_noise | angular_error_deg_mean | 18.2591 |
| lighting_direction | dinov2_vit_b | flat | angular_error_deg_mean | 20.8126 |
| lighting_direction | dinov2_vit_b | photorealistic | angular_error_deg_mean | 28.4254 |
| lighting_direction | dinov2_vit_b | random_noise | angular_error_deg_mean | 32.9593 |
| lighting_intensity | clip_vit_b16 | flat | mae_mean | 0.7356 |
| lighting_intensity | clip_vit_b16 | photorealistic | mae_mean | 0.6995 |
| lighting_intensity | clip_vit_b16 | random_noise | mae_mean | 0.6332 |
| lighting_intensity | clip_vit_l14 | flat | mae_mean | 0.3598 |
| lighting_intensity | clip_vit_l14 | photorealistic | mae_mean | 0.3642 |
| lighting_intensity | clip_vit_l14 | random_noise | mae_mean | 0.1791 |
| lighting_intensity | dinov2_vit_b | flat | mae_mean | 0.7049 |
| lighting_intensity | dinov2_vit_b | photorealistic | mae_mean | 0.8802 |
| lighting_intensity | dinov2_vit_b | random_noise | mae_mean | 0.4484 |
| relative_depth_regions | clip_vit_b16 | flat | valid_pair_accuracy | 0.6250 |
| relative_depth_regions | clip_vit_b16 | photorealistic | valid_pair_accuracy | 0.6250 |
| relative_depth_regions | clip_vit_b16 | random_noise | valid_pair_accuracy | 0.6250 |
| relative_depth_regions | clip_vit_l14 | flat | valid_pair_accuracy | 0.3750 |
| relative_depth_regions | clip_vit_l14 | photorealistic | valid_pair_accuracy | 0.3125 |
| relative_depth_regions | clip_vit_l14 | random_noise | valid_pair_accuracy | 0.3750 |
| relative_depth_regions | dinov2_vit_b | flat | valid_pair_accuracy | 0.2500 |
| relative_depth_regions | dinov2_vit_b | photorealistic | valid_pair_accuracy | 0.2500 |
| relative_depth_regions | dinov2_vit_b | random_noise | valid_pair_accuracy | 0.1250 |
| surface_normal_aggregate | clip_vit_b16 | flat | angular_error_deg_mean | 88.4717 |
| surface_normal_aggregate | clip_vit_b16 | photorealistic | angular_error_deg_mean | 97.3560 |
| surface_normal_aggregate | clip_vit_b16 | random_noise | angular_error_deg_mean | 66.4175 |
| surface_normal_aggregate | clip_vit_l14 | flat | angular_error_deg_mean | 133.1458 |
| surface_normal_aggregate | clip_vit_l14 | photorealistic | angular_error_deg_mean | 111.0735 |
| surface_normal_aggregate | clip_vit_l14 | random_noise | angular_error_deg_mean | 124.2642 |
| surface_normal_aggregate | dinov2_vit_b | flat | angular_error_deg_mean | 113.7318 |
| surface_normal_aggregate | dinov2_vit_b | photorealistic | angular_error_deg_mean | 121.5009 |
| surface_normal_aggregate | dinov2_vit_b | random_noise | angular_error_deg_mean | 73.0775 |
| viewpoint | clip_vit_b16 | flat | viewpoint_angular_error_deg_mean | 25.8698 |
| viewpoint | clip_vit_b16 | photorealistic | viewpoint_angular_error_deg_mean | 25.2760 |
| viewpoint | clip_vit_b16 | random_noise | viewpoint_angular_error_deg_mean | 30.2381 |
| viewpoint | clip_vit_l14 | flat | viewpoint_angular_error_deg_mean | 91.2478 |
| viewpoint | clip_vit_l14 | photorealistic | viewpoint_angular_error_deg_mean | 88.5635 |
| viewpoint | clip_vit_l14 | random_noise | viewpoint_angular_error_deg_mean | 34.8310 |
| viewpoint | dinov2_vit_b | flat | viewpoint_angular_error_deg_mean | 29.0175 |
| viewpoint | dinov2_vit_b | photorealistic | viewpoint_angular_error_deg_mean | 34.2112 |
| viewpoint | dinov2_vit_b | random_noise | viewpoint_angular_error_deg_mean | 35.1684 |

Full layer-wise and texture-wise results are in `outputs/exp1_full_pipeline_shapenet_run/results/`.

## Failures / Fixes During This Run

1. Initial ShapeNet QC failed with `texture_control_mismatch`.
   - Cause: photorealistic rows imported raw ShapeNet OBJ files while flat/random-noise rows imported normalized GLBs.
   - Fix: `scripts/render_blender.py` now uses the same raw mesh for the entire texture triplet when raw photorealistic material preservation is available.
   - Result after rerender: QC passed 72/72.

2. DINOv2 feature extraction initially failed due Hugging Face cache permissions.
   - Cause: Transformers tried to write under `/Users/jerry/.cache/huggingface`, which sandboxed commands could not create.
   - Fix used: reran DINOv2 extraction with `HF_HOME=data/hf_cache`.
   - Result: DINOv2 ViT-B feature caches were written for final/layer4/layer8/layer12.

3. Objaverse was not included in this local cross-check run.
   - `data/objaverse`, `data/objaverse_cache`, and `data/exp1/manifests/objaverse_assets.jsonl` were absent.
   - The repo now has `scripts/download_objaverse_assets.py` for a bounded Objaverse manifest, but this run used local ShapeNet only.

## Manual Verification Guide

Inspect manifests and QC:

```bash
.venv/bin/python -c "import pandas as pd; print(pd.read_parquet('data/exp1_full_pipeline_shapenet_run/manifests/render_valid.parquet').head())"
.venv/bin/python -c "import pandas as pd; print(pd.read_parquet('data/exp1_full_pipeline_shapenet_run/manifests/render_qc.parquet')['qc_pass'].value_counts())"
```

Inspect renders and contact sheet:

```bash
open data/exp1_full_pipeline_shapenet_run/qc/contact_sheet.png
open data/exp1_full_pipeline_shapenet_run/renders
```

Inspect feature tensors:

```bash
.venv/bin/python -c "import numpy as np; d=np.load('data/exp1_full_pipeline_shapenet_run/features/dinov2_vit_b/final.npz'); print(d['features'].shape, d['render_ids'][:3])"
```

Inspect results:

```bash
.venv/bin/python -c "import pandas as pd; print(pd.read_csv('outputs/exp1_full_pipeline_shapenet_run/results/exp1_results_long.csv').head())"
ls outputs/exp1_full_pipeline_shapenet_run/figures
```

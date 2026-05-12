# Experiment 1 Full Pipeline Run

## Overall Status

**Partially completed: reduced local ModelNet40 run.** The pipeline ran end-to-end from local raw `.off` meshes through normalized `.glb` assets, Blender renders, QC, labels, CLIP feature extraction, linear probe training, aggregation, bootstrap tables, and figures.

This was **not a full Experiment 1 dataset run**. It used 3 local ModelNet40 airplane objects and a small controlled render grid. CLIP ViT-B/16 and CLIP ViT-L/14 completed. DINOv2 ViT-B did not complete because the current Torch Hub `dinov2` code is incompatible with the repo's Python 3.9 environment.

## Environment

- OS: macOS/Darwin 23.6.0 arm64.
- Project venv Python: 3.9.6 at `.venv/bin/python`.
- System `python3`: 3.14.2.
- CUDA: unavailable.
- PyTorch MPS: unavailable in this venv.
- Blender: 5.1.1 at `/Applications/Blender.app/Contents/MacOS/Blender`; not on `PATH`.
- Key packages: `torch==2.8.0`, `transformers==4.57.6`, `open_clip==3.3.0`, `pandas==2.3.3`, `numpy==2.0.2`, `Pillow==11.3.0`, `OmegaConf==2.3.0`, `matplotlib==3.9.4`.

## Commands Run

Documentation/setup inspection included `rg --files`, `git status --short`, `ls -la`, `find` for Experiment 1/proposal files, `sed` reads of `AGENTS.md`, `EXPERIMENT1_TASKS.md`, `EXPERIMENT1_HANDOFF.md`, `README.md`, `Makefile`, configs, and pipeline scripts. No proposal PDFs were present in this repo.

Main validation and pipeline commands:

```bash
.venv/bin/python --version
python3 --version
uname -a
which blender
which nvidia-smi
/Applications/Blender.app/Contents/MacOS/Blender --version
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -m pytest tests/test_exp1_manifest.py tests/test_exp1_render_grid.py tests/test_exp1_assets.py tests/test_exp1_labels.py tests/test_exp1_splits.py tests/test_exp1_features.py tests/test_exp1_probe_metrics.py tests/test_exp1_qc.py tests/test_exp1_results.py tests/test_exp1_configs.py tests/test_exp1_pipeline.py

PYTHONPATH=. .venv/bin/python scripts/preprocess_assets.py --config reports/experiment1_full_pipeline_run_config.yaml --modelnet-root data/modelnet40 --max-objects 3 --overwrite
PYTHONPATH=. .venv/bin/python scripts/create_render_plan.py --config reports/experiment1_full_pipeline_run_config.yaml --asset-manifest data/exp1_full_pipeline_run/manifests/assets_normalized.jsonl
PYTHONPATH=. .venv/bin/python scripts/run_exp1_pipeline.py --config reports/experiment1_full_pipeline_run_config.yaml --stages render_chunks
BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender bash data/exp1_full_pipeline_run/manifests/render_chunks/run_blender_chunks.sh
PYTHONPATH=. .venv/bin/python scripts/run_exp1_pipeline.py --config reports/experiment1_full_pipeline_run_config.yaml --stages post_render

PYTHONPATH=. .venv/bin/python scripts/extract_exp1_features.py --config reports/experiment1_full_pipeline_run_config.yaml --render-manifest data/exp1_full_pipeline_run/manifests/render_valid.parquet --feature-dir data/exp1_full_pipeline_run/features --models clip_vit_b16 --layers final layer4 layer8 layer12 --device cpu
PYTHONPATH=. .venv/bin/python scripts/extract_exp1_features.py --config reports/experiment1_full_pipeline_run_config.yaml --render-manifest data/exp1_full_pipeline_run/manifests/render_valid.parquet --feature-dir data/exp1_full_pipeline_run/features --models clip_vit_l14 --layers final layer4 layer8 layer12 --device cpu
PYTHONPATH=. .venv/bin/python scripts/extract_exp1_features.py --config reports/experiment1_full_pipeline_run_config.yaml --render-manifest data/exp1_full_pipeline_run/manifests/render_valid.parquet --feature-dir data/exp1_full_pipeline_run/features --models dinov2_vit_b --layers final layer4 layer8 layer12 --device cpu

PYTHONPATH=. .venv/bin/python scripts/build_labels.py --config reports/experiment1_full_pipeline_run_config.yaml --render-manifest data/exp1_full_pipeline_run/manifests/render_valid.parquet --labels-dir data/exp1_full_pipeline_run/labels
PYTHONPATH=. .venv/bin/python scripts/train_all_exp1_probes.py --config reports/experiment1_full_pipeline_run_config.yaml --models clip_vit_b16 clip_vit_l14 dinov2_vit_b --layers final layer4 layer8 layer12 --tasks surface_normal_aggregate relative_depth_regions camera_distance viewpoint lighting_direction lighting_intensity apparent_scale --device cpu --epochs 3 --batch-size 16
PYTHONPATH=. .venv/bin/python scripts/aggregate_exp1_results.py --config reports/experiment1_full_pipeline_run_config.yaml --probe-root outputs/exp1_full_pipeline_run/probes --output outputs/exp1_full_pipeline_run/results/exp1_results_long.csv --texture-drops-output outputs/exp1_full_pipeline_run/results/exp1_texture_drops.csv --bootstrap-output outputs/exp1_full_pipeline_run/results/exp1_bootstrap_ci.csv
PYTHONPATH=. .venv/bin/python scripts/make_exp1_figures.py --config reports/experiment1_full_pipeline_run_config.yaml --results outputs/exp1_full_pipeline_run/results/exp1_results_long.csv --texture-drops outputs/exp1_full_pipeline_run/results/exp1_texture_drops.csv --output-dir outputs/exp1_full_pipeline_run/figures
```

## Dataset Availability

- `data/modelnet40`: found and used. It contains 12,311 `.off` files across 40 categories.
- `data/shapenet_hf/ShapeNetCore`: exists locally, but was not used in this run.
- `data/objaverse`: not found.
- `data/synthetic_primitives`: only a catalog placeholder existed before this run.

Run subset:

- Dataset used: ModelNet40.
- Objects processed: 3.
- Categories processed: 1 (`airplane`).
- Splits after deterministic object-disjoint reassignment: 1 train object, 1 val object, 1 test object.
- Render rows planned: 72.
- Texture rows planned: 24 each for `photorealistic`, `flat`, `random_noise`.

## Generated Artifacts

- Run config: `reports/experiment1_full_pipeline_run_config.yaml`.
- Raw asset manifest: `data/exp1_full_pipeline_run/manifests/assets.jsonl`.
- Normalized asset manifest: `data/exp1_full_pipeline_run/manifests/assets_normalized.jsonl`.
- Object split manifest: `data/exp1_full_pipeline_run/manifests/object_splits.jsonl`.
- Normalized GLBs: `data/exp1_full_pipeline_run/normalized_assets/modelnet40/.../*.glb`.
- Render plan: `data/exp1_full_pipeline_run/manifests/render_plan.jsonl`.
- Render chunks and shell script: `data/exp1_full_pipeline_run/manifests/render_chunks/`.
- Combined render status: `data/exp1_full_pipeline_run/manifests/render_chunks/render_status.jsonl`.
- Renders: `data/exp1_full_pipeline_run/renders/`.
- QC manifest: `data/exp1_full_pipeline_run/manifests/render_qc.parquet`.
- Valid render manifest: `data/exp1_full_pipeline_run/manifests/render_valid.parquet`.
- Contact sheet: `data/exp1_full_pipeline_run/qc/contact_sheet.png`.
- Labels: `data/exp1_full_pipeline_run/labels/*.parquet`.
- CLIP feature caches: `data/exp1_full_pipeline_run/features/clip_vit_b16/*.npz` and `data/exp1_full_pipeline_run/features/clip_vit_l14/*.npz`.
- Probe outputs: `outputs/exp1_full_pipeline_run/probes/`.
- Results tables: `outputs/exp1_full_pipeline_run/results/exp1_results_long.csv`, `exp1_texture_drops.csv`, `exp1_bootstrap_ci.csv`.
- Figures: `outputs/exp1_full_pipeline_run/figures/` with 172 PNGs.

## Produced Labels

The valid render manifest contains:

- `source_dataset`, `object_id`, `category`, `split`;
- `texture_condition`, `texture_seed`;
- `camera_distance`, `camera_azimuth_deg`, `camera_elevation_deg`, `camera_fov_deg`;
- `light_type`, `light_azimuth_deg`, `light_elevation_deg`, `light_intensity`;
- `object_scale`;
- `rgb_path`, `depth_path`, `normal_path`, `mask_path`;
- `render_status`, QC columns, and texture-control group IDs.

Task label files:

- `labels_surface_normal_aggregate.parquet`: 66 rows, 66 valid.
- `labels_relative_depth_regions.parquet`: 66 rows, 36 valid after reducing the threshold for this tiny airplane subset.
- `labels_camera.parquet`: 66 rows, 66 valid; includes log distance and azimuth/elevation sin/cos labels.
- `labels_lighting.parquet`: 66 rows, 66 valid; includes light direction vector and log intensity.
- `labels_scale.parquet`: 66 rows, 66 valid; includes object scale and foreground mask area fraction.

Note: this reduced grid varies camera distance, azimuth, and lighting intensity. Camera elevation, light direction, and object scale are fixed, so results for elevation/light-direction/object-scale components are only pipeline checks, not scientifically meaningful estimates.

## Validation Checks

Passed:

- Pure Python Experiment 1 tests: 81 passed, 3 skipped.
- Asset preprocessing: 3 assets normalized, 0 failed.
- GLB reloadability: all 3 normalized `.glb` files load with `trimesh`; file sizes were nonzero.
- Render IDs: no duplicates in 72 planned rows.
- Texture controls: all texture-control groups had exactly the three texture conditions and matched camera/light/scale fields.
- Blender rendering: 72 render attempts wrote status rows.
- QC: 66/72 renders passed and were retained; all valid rows have nonempty RGB/depth/normal/mask files.
- Feature caches: CLIP ViT-B/16 and CLIP ViT-L/14 caches are finite and contain `render_ids`.
- Probe training: 168 CLIP probe jobs completed from cached frozen features; DINO jobs skipped due missing feature cache.
- Results: 2,112 long-result rows, 1,360 texture-drop rows, 648 bootstrap rows, and 172 figures were produced.

QC failures:

- 6 train renders failed QC with `foreground_fraction_too_small` at foreground fraction `0.008789`; these rows were excluded downstream.

Feature cache shapes:

| model | layer | shape |
|---|---|---:|
| clip_vit_b16 | final | `(66, 512)` |
| clip_vit_b16 | layer4/layer8/layer12 | `(66, 768)` |
| clip_vit_l14 | final | `(66, 768)` |
| clip_vit_l14 | layer4/layer8/layer12 | `(66, 1024)` |

## Results Summary

These are final-layer test metrics from the reduced run. Lower is better for angular/MAE metrics; higher is better for relative-depth accuracy.

| task | model | texture | metric | test value |
|---|---|---|---|---:|
| surface_normal_aggregate | clip_vit_b16 | flat | angular_error_deg_mean | 98.9733 |
| surface_normal_aggregate | clip_vit_b16 | photorealistic | angular_error_deg_mean | 97.2098 |
| surface_normal_aggregate | clip_vit_b16 | random_noise | angular_error_deg_mean | 96.5516 |
| surface_normal_aggregate | clip_vit_l14 | flat | angular_error_deg_mean | 36.8729 |
| surface_normal_aggregate | clip_vit_l14 | photorealistic | angular_error_deg_mean | 39.2286 |
| surface_normal_aggregate | clip_vit_l14 | random_noise | angular_error_deg_mean | 48.6930 |
| relative_depth_regions | clip_vit_b16 | flat | valid_pair_accuracy | 0.3333 |
| relative_depth_regions | clip_vit_b16 | photorealistic | valid_pair_accuracy | 0.3333 |
| relative_depth_regions | clip_vit_b16 | random_noise | valid_pair_accuracy | 0.3333 |
| relative_depth_regions | clip_vit_l14 | flat | valid_pair_accuracy | 0.6667 |
| relative_depth_regions | clip_vit_l14 | photorealistic | valid_pair_accuracy | 1.0000 |
| relative_depth_regions | clip_vit_l14 | random_noise | valid_pair_accuracy | 1.0000 |
| camera_distance | clip_vit_b16 | flat | mae_mean | 0.1467 |
| camera_distance | clip_vit_b16 | photorealistic | mae_mean | 0.1717 |
| camera_distance | clip_vit_b16 | random_noise | mae_mean | 0.1571 |
| camera_distance | clip_vit_l14 | flat | mae_mean | 0.2635 |
| camera_distance | clip_vit_l14 | photorealistic | mae_mean | 0.2521 |
| camera_distance | clip_vit_l14 | random_noise | mae_mean | 0.1228 |
| viewpoint | clip_vit_b16 | flat | viewpoint_angular_error_deg_mean | 21.8077 |
| viewpoint | clip_vit_b16 | photorealistic | viewpoint_angular_error_deg_mean | 21.0116 |
| viewpoint | clip_vit_b16 | random_noise | viewpoint_angular_error_deg_mean | 25.8443 |
| viewpoint | clip_vit_l14 | flat | viewpoint_angular_error_deg_mean | 47.2990 |
| viewpoint | clip_vit_l14 | photorealistic | viewpoint_angular_error_deg_mean | 41.0609 |
| viewpoint | clip_vit_l14 | random_noise | viewpoint_angular_error_deg_mean | 32.2485 |
| lighting_direction | clip_vit_b16 | flat | angular_error_deg_mean | 31.6241 |
| lighting_direction | clip_vit_b16 | photorealistic | angular_error_deg_mean | 30.6573 |
| lighting_direction | clip_vit_b16 | random_noise | angular_error_deg_mean | 28.8602 |
| lighting_direction | clip_vit_l14 | flat | angular_error_deg_mean | 15.5661 |
| lighting_direction | clip_vit_l14 | photorealistic | angular_error_deg_mean | 14.1712 |
| lighting_direction | clip_vit_l14 | random_noise | angular_error_deg_mean | 23.0306 |
| lighting_intensity | clip_vit_b16 | flat | mae_mean | 0.3228 |
| lighting_intensity | clip_vit_b16 | photorealistic | mae_mean | 0.3622 |
| lighting_intensity | clip_vit_b16 | random_noise | mae_mean | 0.3316 |
| lighting_intensity | clip_vit_l14 | flat | mae_mean | 0.2966 |
| lighting_intensity | clip_vit_l14 | photorealistic | mae_mean | 0.3004 |
| lighting_intensity | clip_vit_l14 | random_noise | mae_mean | 0.1937 |
| apparent_scale | clip_vit_b16 | flat | mae_mean | 0.2848 |
| apparent_scale | clip_vit_b16 | photorealistic | mae_mean | 0.2779 |
| apparent_scale | clip_vit_b16 | random_noise | mae_mean | 0.1959 |
| apparent_scale | clip_vit_l14 | flat | mae_mean | 0.3273 |
| apparent_scale | clip_vit_l14 | photorealistic | mae_mean | 0.3069 |
| apparent_scale | clip_vit_l14 | random_noise | mae_mean | 0.0997 |

Full layer-wise and texture-wise results are in `outputs/exp1_full_pipeline_run/results/`.

## Failures / Blockers

1. Blender crashed inside the sandbox.
   - Command: `BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender bash data/exp1_full_pipeline_run/manifests/render_chunks/run_blender_chunks.sh`
   - Error: segmentation fault in Blender Metal backend initialization.
   - Fix used: reran the same command with approval outside the sandbox; rendering then succeeded.

2. Initial CLIP feature extraction had no network.
   - Command: CLIP ViT-B/16 extraction command above.
   - Error: DNS failure resolving `huggingface.co`; checkpoint not cached.
   - Fix used: reran with approved network access; CLIP B/16 and L/14 completed.

3. DINOv2 ViT-B feature extraction failed.
   - Command: DINOv2 extraction command above.
   - Error: `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` in downloaded `dinov2` Torch Hub code.
   - Likely cause: current Torch Hub `facebookresearch/dinov2` uses Python 3.10 union syntax, while this repo venv is Python 3.9.6.
   - Suggested fix: use a Python >=3.10 venv, pin Torch Hub to a Python-3.9-compatible DINOv2 commit, or replace the wrapper with a `transformers` DINOv2 loader.

4. Reduced-run scientific limitations.
   - Only 3 ModelNet objects were used.
   - All objects are airplanes.
   - Light direction and object scale were fixed.
   - Cross-texture probe mode was disabled in the run config to keep the reduced job small.
   - Results should be treated as pipeline smoke evidence, not research findings.

## Manual Verification Guide

Inspect normalized assets:

```bash
ls -lh data/exp1_full_pipeline_run/normalized_assets/modelnet40/*/airplane/*.glb
.venv/bin/python -c "import trimesh; trimesh.load('data/exp1_full_pipeline_run/normalized_assets/modelnet40/train/airplane/airplane_airplane_0629.glb')"
```

Inspect renders and contact sheet:

```bash
open data/exp1_full_pipeline_run/qc/contact_sheet.png
find data/exp1_full_pipeline_run/renders -name rgb.png | head
```

Inspect metadata and labels:

```bash
.venv/bin/python -c "import pandas as pd; print(pd.read_parquet('data/exp1_full_pipeline_run/manifests/render_valid.parquet').head())"
.venv/bin/python -c "import pandas as pd; print(pd.read_parquet('data/exp1_full_pipeline_run/labels/labels_surface_normal_aggregate.parquet').head())"
```

Inspect feature tensors:

```bash
.venv/bin/python -c "import numpy as np; d=np.load('data/exp1_full_pipeline_run/features/clip_vit_b16/final.npz'); print(d['features'].shape, d['render_ids'][:3])"
```

Inspect probe metrics and aggregate results:

```bash
find outputs/exp1_full_pipeline_run/probes -name metrics.json | head
.venv/bin/python -c "import pandas as pd; print(pd.read_csv('outputs/exp1_full_pipeline_run/results/exp1_results_long.csv').head())"
ls outputs/exp1_full_pipeline_run/figures | head
```

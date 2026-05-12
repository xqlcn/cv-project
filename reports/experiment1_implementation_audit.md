# Experiment 1 Implementation Audit

## Overall Status

Experiment 1 is **partially implemented**. The repository has a real manifest-driven spine, render-plan generation, asset discovery/normalization, Blender rendering code, QC, MVP label builders, frozen feature extraction wrappers, linear probes for the two MVP tasks, result aggregation, figures, and a smoke pipeline runner.

It does **not yet appear complete or fully correct** against the project outline. The biggest blockers are:

- object-disjoint split assignment is not implemented from configured fractions; existing source splits are only preserved;
- full-task probe training is not implemented for camera distance, viewpoint, lighting direction, lighting intensity, or scale;
- configured cross-texture evaluation is not actually implemented;
- HuggingFace CLIP "final" features use the unprojected vision-tower CLS token, not the projected CLIP image embedding;
- Blender random-noise texture uses procedural object-coordinate noise, which is deterministic but questionable for "uncorrelated with object geometry";
- no local end-to-end smoke run through Blender, features, probes, and results was completed during this audit because Blender is not on `PATH` and model weights were not exercised.

## Checked Sources

Project requirements and outline:

- `AGENTS.md`
- `EXPERIMENT1_TASKS.md`
- `EXPERIMENT1_HANDOFF.md`
- `README.md`
- `/Users/jerry/Downloads/project implementatio.pdf` via `pypdf`
- existing generated preprocessing reports under `outputs/sanity_check_preprocessing/metadata/`

Implementation files inspected:

- Configs: `configs/exp1_smoke.yaml`, `configs/exp1_mvp.yaml`, `configs/exp1_full.yaml`, `configs/exp1/paths.yaml`, `configs/exp1/render.yaml`, `configs/exp1/tasks.yaml`
- Metadata/render plan: `exp1/metadata/schema.py`, `exp1/metadata/manifest.py`, `exp1/rendering/grid.py`, `scripts/create_render_plan.py`
- Assets/preprocessing: `exp1/assets/discover.py`, `exp1/assets/normalize.py`, `exp1/assets/validate.py`, `exp1/assets/shapenet.py`, `exp1/assets/preprocessing_sanity.py`, `scripts/preprocess_assets.py`, `scripts/sanity_check_preprocessing.py`
- Blender/rendering: `scripts/render_blender.py`, `blender/mesh_utils.py`, `blender/material_utils.py`, `blender/camera_utils.py`, `blender/lighting_utils.py`, `blender/preflight_render_plan.py`, `exp1/rendering/camera.py`, `exp1/rendering/lighting.py`, `exp1/rendering/materials.py`, `exp1/rendering/passes.py`, `exp1/rendering/qc.py`
- Data/labels: `exp1/data/image_dataset.py`, `exp1/data/feature_dataset.py`, `exp1/data/splits.py`, `exp1/data/label_join.py`, `scripts/build_labels.py`, `exp1/tasks/*.py`
- Models/features/probes/evaluation: `src/models/clip_extractor.py`, `src/models/dino_extractor.py`, `exp1/features/*.py`, `scripts/extract_exp1_features.py`, `exp1/probes/*.py`, `scripts/train_exp1_probe.py`, `scripts/train_all_exp1_probes.py`, `exp1/evaluation/*.py`, `exp1/analysis/plots.py`
- Runner/tests: `scripts/run_exp1_pipeline.py`, `Makefile`, `tests/test_exp1_*.py`, `tests/test_sanity_check_preprocessing.py`

## Completion Checklist

| Requirement | Status | Evidence | How to Verify | Notes / Gaps |
|---|---:|---|---|---|
| Task 1: Experiment 1 configs exist and load | COMPLETE | `configs/exp1_smoke.yaml`, `configs/exp1_mvp.yaml`, `configs/exp1_full.yaml`, groups under `configs/exp1/`; tests in `tests/test_exp1_configs.py` | `python3 -m pytest tests/test_exp1_configs.py` | MVP config enables CLIP B/16 and DINOv2 B only; CLIP L/14 is only in full config. |
| Configs require no local absolute paths | COMPLETE | `tests/test_exp1_configs.py` checks no `/Users/`; paths use `${paths.project_root}` and env | same test | Runtime defaults enable ShapeNet/Objaverse sources whose local dirs may not exist. |
| Task 2: Manifest schema, deterministic `render_id`, save/load, validation | COMPLETE | `exp1/metadata/schema.py::generate_render_id`; `exp1/metadata/manifest.py::validate_render_manifest` | `python3 -m pytest tests/test_exp1_manifest.py` | Does not validate physical output files; QC handles that later. |
| Task 3: Synthetic render-plan generator | COMPLETE | `exp1/rendering/grid.py::build_render_plan`; `scripts/create_render_plan.py`; tests verify texture triplets and ID uniqueness | `python3 -m pytest tests/test_exp1_render_grid.py` | Generates full Cartesian grid, not stratified sampling. |
| Render-plan texture-only controls match camera/light/scale | COMPLETE | `texture_control_group_id` generated in `build_render_plan`; tests assert non-texture fields match | inspect `data/exp1_sanity/render_plan.jsonl` or run test | Good foundation for texture controls. |
| Task 4: Asset discovery supports ShapeNet/Objaverse/ModelNet/custom meshes | PARTIAL | `scripts/preprocess_assets.py`; `exp1/assets/shapenet.py`; `exp1/assets/discover.py` | `python3 -m pytest tests/test_exp1_assets.py` | Supports local/HF ShapeNet snapshots and Objaverse dirs, but no automatic Objaverse API/download path. |
| Task 4: Normalize meshes to centered unit-scale GLB | COMPLETE | `exp1/assets/normalize.py::normalize_asset_record` writes `.glb`; tests reload and check centered max extent | `python3 -m pytest tests/test_exp1_assets.py::test_normalize_asset_manifest_writes_centered_unit_glb` | Orientation/canonical up-axis is not normalized. |
| Task 4: Invalid meshes are marked rather than crashing | COMPLETE | `normalize_asset_record` catches exceptions and sets `asset_status="failed"` | `python3 -m pytest tests/test_exp1_assets.py::test_invalid_mesh_is_marked_without_crashing` | `--fail-fast` can opt into crashing. |
| Task 4: Object-disjoint split file is created | PARTIAL | `exp1/assets/validate.py::write_object_split_manifest` writes one row per object | inspect `data/exp1/manifests/object_splits.jsonl` | It preserves existing splits only. It does not assign train/val/test from `configs/exp1/paths.yaml` fractions. ShapeNet with no split markers defaults to all `train`. |
| Object-disjoint train/val/test assignment from split fractions | BUG / LIKELY INCORRECT | `splits.fractions` exists in config, but `rg fractions exp1 scripts` finds no assignment code | `rg -n "fractions|random.*split" exp1 scripts` | This blocks valid probe evaluation on sources without existing splits. |
| Task 5: Blender renderer skeleton | PARTIAL | `scripts/render_blender.py::render_record` writes RGB/depth/normal/mask/meta; actual Blender integration tests are skipped if `blender` missing | `python3 -m pytest tests/test_exp1_blender_renderer.py`; then run a Blender chunk manually | Current environment: 8 Blender tests passed, 2 render tests skipped because Blender is not on `PATH`. Existing ignored artifacts show prior successful renders. |
| Task 5: Geometry buffers have expected shapes | PARTIAL | `exp1/rendering/passes.py::render_geometry_buffers`; tests cover shape only when Blender is available | `blender --background --python scripts/render_blender.py -- --record-file ...` | Ray-cast normals/depth are plausible, but coordinate convention should be manually checked with a known cube/sphere. |
| Task 6: Photorealistic/original material preservation | PARTIAL | `exp1/rendering/materials.py::apply_materials` preserves imported Blender materials when `has_preservable_material`; records fallback metadata | inspect `render_meta.json` or status JSONL for `photorealistic_material_status` | Main renderer imports `normalized_mesh_path`; normalization via trimesh can strip or alter original materials, causing fallback even if raw source had textures. |
| Task 6: Flat textureless material with lighting/shading | COMPLETE | `apply_materials` calls `assign_principled_material(... texture_type="flat", preserve_existing=False)`; `textures.flat.use_emission=false` in config | inspect flat `rgb.png` and `render_meta.json` | Uses constant albedo and keeps lighting. |
| Task 6: Random-noise texture deterministic and uncorrelated | PARTIAL / QUESTIONABLE | `blender/material_utils.py` uses `ShaderNodeTexNoise` seeded by `texture_seed` and Object coordinates; metadata says `procedural_noise` | inspect `random_noise_seed` and `random_noise_texture_type` in `render_meta.json` | Deterministic, but not an independent random image texture; object-coordinate procedural noise can vary with geometry/scale and has no saved `random_texture_path`. |
| Task 6: Texture-control geometry buffers match across conditions | PARTIAL | `exp1/rendering/qc.py::apply_texture_control_qc`; test exists; Blender integration skipped locally | `PYTHONPATH=. python3 scripts/validate_renders.py --input <status.jsonl>` | Implemented in QC, but needs actual rendered triplets for verification. |
| Task 7: Render QC and contact sheets | PARTIAL | `scripts/validate_renders.py`, `scripts/make_contact_sheet.py`, `exp1/rendering/qc.py` | `python3 -m pytest tests/test_exp1_qc.py` | Direct `validate_renders.py` defaults to render plan rows with `render_status=pending`; use combined status JSONL or runner stage. |
| Task 8: Surface-normal aggregate labels | COMPLETE | `exp1/tasks/surface_normals.py::build_surface_normal_aggregate_labels` | `python3 -m pytest tests/test_exp1_labels.py` | Uses camera-space normal path and mask. |
| Task 8: Relative-depth region labels | COMPLETE | `exp1/tasks/relative_depth.py::build_relative_depth_labels`; 3x3 grid and region pairs in config | `python3 -m pytest tests/test_exp1_labels.py` | Uses smaller positive camera-space depth as closer. |
| Full task labels: camera distance/viewpoint/lighting/scale | PARTIAL | `exp1/tasks/camera.py`, `lighting.py`, `scale.py`; `scripts/build_labels.py` writes grouped `labels_camera`, `labels_lighting`, `labels_scale` | `PYTHONPATH=. python3 scripts/build_labels.py --tasks camera_distance viewpoint lighting_direction lighting_intensity apparent_scale ...` | Config task definitions lack `label_path` for these tasks, so probe scripts skip/fail them. |
| Task 9: Dataset and split utilities | PARTIAL | `exp1/data/image_dataset.py`, `feature_dataset.py`, `splits.py`, `label_join.py`; tests cover filtering/join order | `python3 -m pytest tests/test_exp1_splits.py` | Split assertion exists, but split creation from object list does not. |
| Joins use `render_id`, not row order | COMPLETE | `exp1/data/label_join.py::join_labels_by_render_id`; tests shuffle label rows | same test | Good. |
| Task 10: Feature extraction for CLIP/DINOv2 final/intermediate layers | PARTIAL | `exp1/features/extract.py::build_backbone_extractor`; model configs include CLIP B/16, CLIP L/14, DINOv2 B/L | `PYTHONPATH=. python3 scripts/extract_exp1_features.py --models clip_vit_b16 --layers final --limit 8 --device cpu` | Not locally run against real weights; requires cached/downloadable model weights. |
| Frozen-backbone behavior | COMPLETE | `src/models/clip_extractor.py` and `src/models/dino_extractor.py` set `requires_grad=False`, `eval()`, and use `torch.inference_mode()` | inspect lines around `requires_grad=False`; run a small extractor once weights are available | Probe training consumes cached NumPy features, so backbone cannot update there. |
| CLIP final feature is language-aligned projection | BUG / LIKELY INCORRECT | HF path loads `CLIPVisionModelWithProjection` then discards projection via `self.hf_model = vm.vision_model`; final uses last hidden CLS | inspect `src/models/clip_extractor.py:55-95` | For the stated CLIP final/language-aligned question, this should likely use `image_embeds` or apply `visual_projection`. OpenCLIP path does use `encode_image`. |
| Intermediate layer probing | PARTIAL | `parse_layer_name`, `requested_layer_numbers`, CLIP hidden states, DINO get_intermediate_layers | `PYTHONPATH=. python3 scripts/extract_exp1_features.py --layers layer4 layer8 layer12 final ...` | Hook/index semantics should be checked per model; no integration test with real CLIP/DINO weights. |
| Task 11: Linear probes for MVP tasks | COMPLETE | `exp1/probes/train.py`, `heads.py`, `losses.py`, `metrics.py`; synthetic overfit tests pass | `python3 -m pytest tests/test_exp1_probe_metrics.py` | Linear head optionally includes LayerNorm. |
| Task 11: Probes for full regression tasks | MISSING | `_loss_for_task`, `_metrics_for_task`, `_score`, `_prediction_dataframe` only support `surface_normal_aggregate` and `relative_depth_regions` | `rg -n "Unsupported Experiment 1 probe task" exp1/probes/train.py` | Full tasks in `exp1_full.yaml` are not trainable with current probe code. |
| Within-texture and cross-texture training modes | PARTIAL / MISSING | CLI can filter all splits to one texture via `--texture-condition`; no train-texture/test-texture split matrix | `rg -n "cross_texture|train_texture|test_texture" exp1 scripts` | Config contains `cross_texture: true`, but code does not implement cross-texture generalization. |
| Task 12: Long-format aggregation by task/model/layer/texture | COMPLETE | `exp1/evaluation/metrics.py::aggregate_probe_metrics` | `python3 -m pytest tests/test_exp1_results.py` | Depends on probe metadata. |
| Texture-dependence drops and layer-wise plots | PARTIAL | `compute_texture_dependence_drops`, `plot_layerwise_metrics`, `plot_texture_drops` | `PYTHONPATH=. python3 scripts/aggregate_exp1_results.py --allow-empty`; `python3 -m pytest tests/test_exp1_results.py` | No cross-texture heatmaps. |
| Object-level bootstrap confidence intervals | PARTIAL / MISSING | Helper `bootstrap_mean_ci(values, units=...)` exists | test: `tests/test_exp1_results.py::test_bootstrap_mean_ci_can_resample_by_object` | Not integrated into aggregation from prediction files; no CI output table is produced. |
| Task 13: Pipeline runner and Makefile | PARTIAL | `scripts/run_exp1_pipeline.py`, `Makefile` targets `exp1-smoke`, `exp1-smoke-post` | `PYTHONPATH=. python3 scripts/run_exp1_pipeline.py --config configs/exp1_smoke.yaml --stages smoke_prepare --dry-run` | Smoke prepare works as a plan/chunk generator; complete post-render+ML path requires Blender and model weights. |
| Generated outputs ignored by Git | COMPLETE | `.gitignore` ignores `data`, `outputs`, `data/exp1/`, `outputs/exp1/` | `git check-ignore data/exp1/foo outputs/exp1/foo` | Source docs `AGENTS.md`, `EXPERIMENT1_TASKS.md`, and `EXPERIMENT1_HANDOFF.md` are also ignored in this repo. |

## Data and Label Outputs

| Artifact / Label | Producer | Output Path | Consumer | Verification |
|---|---|---|---|---|
| Raw asset manifest | `scripts/preprocess_assets.py` | `data/exp1/manifests/assets.jsonl` | preprocessing audit, normalization | `head data/exp1/manifests/assets.jsonl` |
| Normalized asset manifest | `scripts/preprocess_assets.py`, `exp1/assets/normalize.py` | `data/exp1/manifests/assets_normalized.jsonl` | `scripts/create_render_plan.py` | check `normalized_mesh_path` files exist |
| Normalized GLBs | `exp1/assets/normalize.py::normalize_asset_record` | `data/exp1/normalized_assets/<dataset>/<split>/<category>/<object_id>.glb` | Blender renderer | open in Blender or run `blender/preflight_render_plan.py` |
| Object split manifest | `exp1/assets/validate.py::write_object_split_manifest` | `data/exp1/manifests/object_splits.jsonl` | manual leakage checks | verify each `object_id` appears once |
| Optional physical texture-condition GLBs | `scripts/sanity_check_preprocessing.py` | `outputs/sanity_check_preprocessing/glbs/.../{photorealistic,flat,random_noise}.glb` | manual inspection only | inspect `outputs/sanity_check_preprocessing/metadata/preprocessing_report.md` |
| Render plan | `scripts/create_render_plan.py` | `data/exp1/manifests/render_plan.jsonl`; config also has `render_plan.parquet` | render chunks, Blender | count textures and `texture_control_group_id` groups |
| Render chunks | `scripts/run_exp1_pipeline.py::write_render_chunks` | `data/exp1/manifests/render_chunks/chunk_*.jsonl` | Blender shell script | inspect generated `run_blender_chunks.sh` |
| Render status manifest | `scripts/render_blender.py`, combined by `scripts/run_exp1_pipeline.py` | `data/exp1/manifests/render_chunks/render_status.jsonl` | QC/labels | all successful rows should have `render_status=success` |
| RGB/depth/normal/mask render outputs | `scripts/render_blender.py::render_record` | `rgb.png`, `depth.npy`, `normal_camera.npy`, `mask.npy` under each render folder | QC and label builders | load arrays and inspect shapes |
| Render metadata | `scripts/render_blender.py::render_record` | `render_meta.json` beside render outputs | manual audit/debugging | inspect camera/light/material fields |
| QC manifest | `scripts/validate_renders.py` | `data/exp1/manifests/render_qc.parquet` | contact sheet, valid manifest | check `qc_pass`, `qc_error_message` |
| Valid render manifest | `scripts/validate_renders.py` | `data/exp1/manifests/render_valid.parquet` | labels/features/probes | ensure no failed renders remain |
| Contact sheet | `scripts/make_contact_sheet.py` | `data/exp1/qc/contact_sheet.png` | manual visual QC | open PNG |
| Surface-normal aggregate labels | `scripts/build_labels.py` | `data/exp1/labels/labels_surface_normal_aggregate.parquet` | `scripts/train_exp1_probe.py` | check `label_valid` and mean normal columns |
| Relative-depth labels | `scripts/build_labels.py` | `data/exp1/labels/labels_relative_depth_regions.parquet` | `scripts/train_exp1_probe.py` | check `pair_*_label`, `pair_*_valid` |
| Camera/viewpoint labels | `scripts/build_labels.py` | `data/exp1/labels/labels_camera.parquet` | not wired to probe config | check `log_camera_distance`, sin/cos columns |
| Lighting labels | `scripts/build_labels.py` | `data/exp1/labels/labels_lighting.parquet` | not wired to probe config | check `light_dir_*`, `log_light_intensity` |
| Scale labels | `scripts/build_labels.py` | `data/exp1/labels/labels_scale.parquet` | not wired to probe config | check `object_scale`, `foreground_area_fraction` |
| Feature cache | `scripts/extract_exp1_features.py` | `data/exp1/features/<model>/<layer>.npz` | probe training | NPZ must contain `render_ids` and `features` |
| Probe artifacts | `scripts/train_exp1_probe.py` | `outputs/exp1/probes/<model>/<layer>/<task>/...` | aggregation | inspect `metrics.json`, `predictions.csv` |
| Results tables | `scripts/aggregate_exp1_results.py` | `outputs/exp1/results/exp1_results_long.csv`, `exp1_texture_drops.csv` | figures/report | group by task/model/layer/texture |
| Figures | `scripts/make_exp1_figures.py` | `outputs/exp1/figures/*.png` | manual reporting | open generated PNGs |

## Texture Condition Verification

### `photorealistic`

Implemented in `exp1/rendering/materials.py::apply_materials`. For each mesh object, the code preserves imported materials only if:

- `textures.photorealistic.preserve_imported_materials` is true;
- the manifest/source does not trigger `photorealistic_fallback_reason`;
- Blender import reports a preservable material via `blender/material_utils.py::has_preservable_material`.

Fallback metadata is written as:

- `material_status`;
- `photorealistic_material_status`;
- `photorealistic_fallback_reason`;
- `photorealistic_fallback_color_rgb`.

Manual inspection:

- rendered images: `data/exp1/renders/shapenet_three_textures/.../photorealistic/rgb.png`
- metadata/status: `data/exp1/renders/shapenet_three_textures/status.jsonl`
- physical GLB variants from preprocessing sanity: `outputs/sanity_check_preprocessing/glbs/.../photorealistic.glb`

Concern: the main renderer loads `normalized_mesh_path` first. If normalization/export strips original materials, the photorealistic condition may silently become fallback, though the fallback is recorded.

### `flat`

Implemented in `exp1/rendering/materials.py::apply_materials`, using `assign_principled_material(texture_type="flat", preserve_existing=False)`. Config has `textures.flat.use_emission: false`, and the Blender helper uses Principled BSDF with roughness, so lighting/shading remains active.

Manual inspection:

- rendered images: `data/exp1/renders/shapenet_three_textures/.../flat/rgb.png`
- metadata/status: `flat_color_rgb`, `material_status=flat_override`
- GLB variants: `outputs/sanity_check_preprocessing/glbs/.../flat.glb`

### `random_noise`

Implemented in `exp1/rendering/materials.py` and `blender/material_utils.py`. It is deterministic from `texture_seed`, and metadata records:

- `random_noise_seed`;
- `random_noise_texture_type=procedural_noise`;
- `texture_seed_used`.

Concern: the Blender render path uses a procedural noise node driven through Object coordinates, not a saved random image texture. The PDF recommends generated image textures because they are easier to keep independent of object/category/geometry. This implementation is seedable and not category-coded, but object-coordinate procedural noise can couple texture appearance to geometry, scale, or object coordinates. If the experiment needs a strict no-geometry-leakage random texture, this should be replaced or complemented with a saved random image texture and explicit mapping metadata.

Manual inspection:

- rendered images: `data/exp1/renders/shapenet_three_textures/.../random_noise/rgb.png`
- metadata/status: `random_noise_seed`, `random_noise_texture_type`
- GLB variants: `outputs/sanity_check_preprocessing/glbs/.../random_noise.glb`

## Rendering Metadata Verification

Render-plan generation in `exp1/rendering/grid.py::build_render_plan` records:

- `camera_distance`;
- `camera_azimuth_deg`;
- `camera_elevation_deg`;
- `camera_fov_deg`;
- `object_scale`;
- `light_type`;
- `light_azimuth_deg`;
- `light_elevation_deg`;
- `light_intensity`;
- `texture_seed`;
- `render_seed`;
- `texture_control_group_id`.

Blender rendering in `scripts/render_blender.py::render_record` preserves those fields and adds:

- output paths;
- `normal_coordinate_frame=camera`;
- `depth_convention=positive camera-space z distance; NaN background`;
- `mask_convention`;
- `blender_version`;
- `buffer_shapes`;
- camera location and rotation quaternion;
- material metadata;
- fill light intensity.

Passing into Blender:

- camera setup: `exp1/rendering/camera.py::setup_camera`;
- lighting setup: `exp1/rendering/lighting.py::setup_lighting`;
- object scale: `scripts/render_blender.py::_prepare_scene`.

Recoverability for probe training:

- metadata labels for camera/viewpoint/light/scale are produced by `scripts/build_labels.py`;
- however, full task labels are not wired to `configs/exp1/tasks.yaml` via `label_path`, and probe training does not support these tasks yet.

## Probe / Model Verification

Frozen behavior is mostly sound:

- CLIP and DINO wrappers set all model parameters `requires_grad=False`;
- wrappers call `eval()`;
- forward/extract methods use `torch.inference_mode()`;
- probe training uses cached `.npz` arrays, so no backbone parameters are present during probe optimization.

Supported models/layers:

- definitions exist for `clip_vit_b16`, `clip_vit_l14`, `dinov2_vit_b`, and `dinov2_vit_l`;
- full config enables `clip_vit_b16`, `clip_vit_l14`, and `dinov2_vit_b`;
- MVP config omits `clip_vit_l14`;
- intermediate layer names `layer4`, `layer8`, `layer12` are accepted.

Important model caveat:

- HuggingFace CLIP code discards the projection head by storing `vm.vision_model` and using last hidden-state CLS. This is not clearly the final language-aligned CLIP image representation required by the research question. The OpenCLIP branch uses `encode_image`, but config currently sets `use_open_clip: false`.

Probe support:

- `surface_normal_aggregate`: implemented with linear head, cosine/MSE losses, angular error.
- `relative_depth_regions`: implemented with linear multi-output logits, masked BCE, valid-pair accuracy and balanced accuracy.
- `camera_distance`, `viewpoint`, `lighting_direction`, `lighting_intensity`, `apparent_scale`: labels exist or can be built, but probe training raises unsupported-task errors or skips missing label paths.

Cross-texture:

- training can be filtered to one or more textures for all splits using `--texture-condition`;
- there is no train-on-texture-A/test-on-texture-B matrix despite config fields for cross-texture evaluation.

## Manual Verification Guide

### 1. Run the lightweight test suite

```bash
python3 -m pytest \
  tests/test_exp1_manifest.py \
  tests/test_exp1_render_grid.py \
  tests/test_exp1_assets.py \
  tests/test_exp1_labels.py \
  tests/test_exp1_splits.py \
  tests/test_exp1_features.py \
  tests/test_exp1_probe_metrics.py \
  tests/test_exp1_qc.py \
  tests/test_exp1_results.py \
  tests/test_exp1_configs.py \
  tests/test_sanity_check_preprocessing.py

python3 -m pytest tests/test_exp1_blender_renderer.py
```

Audit result: the first command passed 74 tests. The Blender test file passed 8 tests and skipped 2 actual render integration tests because `blender` was not on `PATH`.

### 2. Preprocessing sanity check

Use local dataset paths only; this does not download data.

```bash
PYTHONPATH=. python3 scripts/sanity_check_preprocessing.py \
  --shapenetcore-root data/shapenet_hf/ShapeNetCore \
  --objaverse-root data/objaverse \
  --output-dir outputs/sanity_check_preprocessing \
  --num-objects-per-dataset 2 \
  --texture-conditions photorealistic flat random_noise \
  --seed 11
```

Inspect:

```bash
open outputs/sanity_check_preprocessing/metadata/preprocessing_report.md
find outputs/sanity_check_preprocessing/glbs -name '*.glb' | head
```

### 3. Check split assignment and leakage

```bash
PYTHONPATH=. python3 - <<'PY'
import pandas as pd
from pathlib import Path
from exp1.assets.validate import assert_object_disjoint_splits

path = Path("data/exp1/manifests/assets_normalized.jsonl")
df = pd.read_json(path, lines=True)
print(df["split"].value_counts(dropna=False))
assert_object_disjoint_splits(df)
print("object-disjoint check passed")
PY
```

If all rows are `train`, the source is not suitable for probe evaluation until a real object-disjoint split assignment is created.

### 4. Create a tiny render plan and chunks

```bash
make exp1-smoke
```

or dry-run:

```bash
PYTHONPATH=. python3 scripts/run_exp1_pipeline.py \
  --config configs/exp1_smoke.yaml \
  --stages smoke_prepare \
  --dry-run
```

Inspect:

```bash
head -3 data/exp1/manifests/render_plan.jsonl
python3 - <<'PY'
import pandas as pd
df = pd.read_json("data/exp1/manifests/render_plan.jsonl", lines=True)
print(df["texture_condition"].value_counts())
print(df.groupby("texture_control_group_id")["texture_condition"].nunique().value_counts())
print(df["split"].value_counts())
PY
```

### 5. Blender preflight and tiny render

```bash
blender --background --python blender/preflight_render_plan.py -- \
  data/exp1/manifests/render_plan.jsonl \
  --project-root "$PWD" \
  --max-imports 3

bash data/exp1/manifests/render_chunks/run_blender_chunks.sh
```

Inspect one triplet:

```bash
find data/exp1/renders -name render_meta.json | head
find data/exp1/renders -name rgb.png | head
```

### 6. QC, labels, contact sheet

```bash
PYTHONPATH=. python3 scripts/run_exp1_pipeline.py \
  --config configs/exp1_smoke.yaml \
  --stages post_render \
  --force
```

Then inspect:

```bash
python3 - <<'PY'
import pandas as pd
for path in [
    "data/exp1/manifests/render_qc.parquet",
    "data/exp1/manifests/render_valid.parquet",
    "data/exp1/labels/labels_surface_normal_aggregate.parquet",
]:
    df = pd.read_parquet(path)
    print(path, df.shape)
    print(df.head())
PY
```

### 7. Feature extraction on a tiny subset

This may require cached or downloadable HuggingFace/torch.hub model weights.

```bash
PYTHONPATH=. python3 scripts/extract_exp1_features.py \
  --config configs/exp1_smoke.yaml \
  --render-manifest data/exp1/manifests/render_valid.parquet \
  --models clip_vit_b16 \
  --layers final \
  --limit 8 \
  --device cpu
```

Verify:

```bash
python3 - <<'PY'
import numpy as np
p = "data/exp1/features/clip_vit_b16/final.npz"
d = np.load(p)
print(d["render_ids"].shape, d["features"].shape, d["features"].dtype)
print("finite:", np.isfinite(d["features"]).all())
PY
```

### 8. Minimal probe

```bash
PYTHONPATH=. python3 scripts/train_exp1_probe.py \
  --config configs/exp1_smoke.yaml \
  --task surface_normal_aggregate \
  --model clip_vit_b16 \
  --layer final \
  --device cpu \
  --epochs 1
```

Verify:

```bash
cat outputs/exp1/probes/clip_vit_b16/final/surface_normal_aggregate/metrics.json
```

## Bugs / Incomplete Items

1. **Object split assignment is missing.** Configured split fractions are not used. Source splits are preserved, which means ShapeNet/Objaverse assets without split labels can all become `train`. This invalidates train/val/test evaluation until fixed.

2. **Full-task probes are missing.** The full config enables camera distance, viewpoint, lighting direction, lighting intensity, and apparent scale, but `exp1/probes/train.py` only supports `surface_normal_aggregate` and `relative_depth_regions`.

3. **Cross-texture evaluation is missing.** Config says cross-texture evaluation is enabled, but the training scripts only filter all splits by texture. There is no train-texture/test-texture matrix.

4. **HuggingFace CLIP final features are likely not the intended CLIP final embedding.** The code uses unprojected vision CLS features. For final language-aligned CLIP comparisons, use `CLIPVisionModelWithProjection` outputs or the projection layer.

5. **Random-noise rendering is questionable.** The main Blender path uses procedural Object-coordinate noise and does not save a random image texture. This may not satisfy the strongest "uncorrelated with geometry" requirement.

6. **Photorealistic preservation can be undermined by normalization.** Main rendering uses normalized GLBs; if trimesh export strips materials, the original texture condition becomes fallback. Metadata records fallback, but the photorealistic subset may be smaller than expected.

7. **Smoke pipeline is not a complete scientific smoke test by default.** `configs/exp1_smoke.yaml` only enables CLIP B/16 final and surface-normal labels. The synthetic setup creates train/val but no test split.

8. **Object-level bootstrap CI helper is not integrated.** A helper exists, but aggregation does not read predictions and produce object-level confidence interval tables.

9. **Direct QC defaults can be misleading.** `scripts/validate_renders.py` defaults to the pending render plan unless passed the combined status manifest. The runner handles this better, but manual use can produce all-failed QC.

10. **No verified local model extraction/probe run was present.** Tests use synthetic batch callbacks and tiny arrays; there are no current feature caches or probe metrics in `data/`/`outputs/`.

## Recommended Next Steps

1. Implement deterministic object-disjoint split assignment for assets lacking reliable splits. Write `exp1/data/splits.py` support that consumes `splits.fractions` and updates/writes an object split manifest.

2. Fix CLIP final feature extraction for HuggingFace CLIP by keeping the projection model and saving both `final_projected` and optionally `vision_cls_final` if both are useful.

3. Replace or augment procedural random noise with generated image textures saved under a configured directory, with seed/path/mapping recorded.

4. Add label paths for full tasks in `configs/exp1/tasks.yaml` or split full-task label configs into explicit task definitions.

5. Generalize probe losses/metrics/predictions for scalar regression, vector regression, and angular vector targets.

6. Implement train-texture/test-texture loops in `train_all_exp1_probes.py`, and save metadata columns `train_texture` and `test_texture` separately.

7. Integrate object-level bootstrap CIs by loading prediction files, joining object IDs from the manifest, and writing a CI table.

8. Run a true smoke end-to-end after Blender and model weights are available: render 2-3 objects, build QC/labels, extract CLIP B/16 final features, train one probe, aggregate results, and inspect the contact sheet.

9. Add an automated smoke test that uses fake feature caches after render/label generation so the post-render/probe/report plumbing can be checked without downloading CLIP/DINO weights.

10. Add manual visual QA notes for camera/depth/normal conventions using a cube or sphere with known camera pose.


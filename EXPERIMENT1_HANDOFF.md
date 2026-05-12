# Experiment 1 Codex Handoff

## Purpose

This document gives implementation context for Experiment 1: Textureless Geometry and Rendering Probes. It complements the root `AGENTS.md` and should be read before writing code.

## Scientific Setup

The experiment renders the same 3D objects under controlled conditions. Mesh geometry is held fixed while rendering variables are changed. The key variable is texture condition:

- `photorealistic`: original/realistic texture where available;
- `flat`: constant-color material with lighting/shading preserved;
- `random_noise`: deterministic random texture uncorrelated with category or geometry.

Additional controlled factors:

- camera distance;
- camera azimuth;
- camera elevation;
- lighting direction;
- lighting intensity;
- object scale/apparent size.

The main comparison is CLIP vs DINOv2 under texture removal/noise. The hypothesis is that DINOv2 should be more robust on geometry-heavy tasks, especially surface normals, relative depth, and viewpoint, while CLIP may perform better in photorealistic settings where geometry correlates with category or semantic priors.

## Minimum Viable Experiment

The MVP should be small but end-to-end:

```text
objects: 10-100 normalized ShapeNet/Objaverse objects
textures: photorealistic, flat, random_noise
render settings: 1 texture-only control pose + several varied poses
models: CLIP ViT-B/16, optionally DINOv2 ViT-B first
layers: final first, then layers 4/8/12
tasks: surface_normal_aggregate, relative_depth_regions
splits: object-disjoint train/val/test
outputs: metrics table + texture/layer breakdown plots
```

Do not start with the full render grid. First prove that the smoke pipeline works.

## Render Output Contract

Each successful render folder should contain:

```text
rgb.png
mask.npy
depth.npy
normal_world.npy or normal_camera.npy
render_meta.json
```

`render_meta.json` should include:

- `render_id`;
- camera intrinsics/extrinsics or enough camera parameters to reconstruct them;
- coordinate frame for normals;
- depth convention and units;
- material status;
- texture seed/path where applicable;
- Blender version if easy to record.

Background pixels in depth/normal outputs should be ignored using `mask.npy`. Prefer `NaN` for invalid depth/background after masking if convenient.

## Manifest Contract

Every stage should use manifests. Required render manifest columns:

```text
render_id
object_id
source_dataset
category
split
raw_mesh_path
normalized_mesh_path
texture_condition
texture_seed
camera_distance
camera_azimuth_deg
camera_elevation_deg
camera_fov_deg
object_scale
light_type
light_azimuth_deg
light_elevation_deg
light_intensity
rgb_path
depth_path
normal_path
mask_path
render_seed
render_status
qc_error_message
```

Optional but recommended columns:

```text
camera_location_x/y/z
light_location_x/y/z
camera_rotation_quat_w/x/y/z
subset
flat_color_rgb
random_texture_path
photorealistic_material_status
qc_foreground_fraction
qc_depth_valid_fraction
qc_normal_mean_norm
```

## Label Files

Keep labels task-specific. Do not overload the render manifest with all derived labels.

Recommended outputs:

```text
data/manifests/labels_surface_normal_aggregate.parquet
data/manifests/labels_relative_depth_regions.parquet
data/manifests/labels_camera.parquet
data/manifests/labels_viewpoint.parquet
data/manifests/labels_lighting.parquet
data/manifests/labels_scale.parquet
```

### Surface normal aggregate

One row per valid render:

```text
render_id
mean_normal_x
mean_normal_y
mean_normal_z
normal_valid_pixel_count
```

Normalize the mean vector. Store/report if too few valid pixels exist.

### Relative depth regions

Either long form:

```text
render_id
pair_id
region_a
region_b
depth_a
depth_b
label
valid
```

or wide form for training:

```text
render_id
pair_0_label ... pair_N_label
pair_0_valid ... pair_N_valid
```

Use smaller camera-space depth as closer. Add a minimum margin to avoid ambiguous labels.

## Feature Cache Contract

Feature files should be keyed by model and layer. Each feature cache must include `render_ids`.

Example:

```text
data/features/exp1/clip_vit_b16/final.npz
data/features/exp1/clip_vit_b16/layer4.npz
data/features/exp1/clip_vit_b16/layer8.npz
data/features/exp1/clip_vit_b16/layer12.npz
data/features/exp1/dinov2_vit_b/final.npz
```

Each `.npz` should contain:

```text
render_ids: array[str]
features: float32 array [N, D]
model_name: optional string/metadata
layer_name: optional string/metadata
```

If using separate metadata JSON, ensure it records the model checkpoint, preprocessing, layer, feature type, and config hash/path.

## Probe Training Contract

Core probes must be linear or very lightweight.

### Surface normal

```text
input: feature [D]
target: mean normal [3]
head: Linear(D, 3)
loss: MSE on normalized predicted vector or cosine/angular loss
metric: angular error in degrees
```

### Relative depth

```text
input: feature [D]
target: multi-output binary labels [num_pairs]
valid_mask: [num_pairs]
head: Linear(D, num_pairs)
loss: BCE-with-logits over valid entries only
metric: valid-pair accuracy, balanced accuracy, per-pair accuracy
```

### Camera/light regression

Use log targets when raw values span multiplicative ranges.

### Viewpoint/lighting direction

Prefer sin/cos or unit-vector targets to avoid angle wraparound issues. For binned classification, report class balance and chance baseline.

## Evaluation Questions

Every final result should be able to answer:

1. Does performance drop from `photorealistic` to `flat` or `random_noise`?
2. Is the drop larger for CLIP than DINOv2?
3. Are geometry-heavy tasks harder for CLIP than appearance/rendering tasks?
4. Do early/intermediate CLIP layers outperform final CLIP layers for geometry tasks?
5. Does cross-texture generalization fail, suggesting texture-specific shortcuts?
6. Are results stable under object-level bootstrap confidence intervals?

## Recommended Smoke Pipeline

Start with a tiny config:

```text
3 objects
3 texture conditions
1 fixed texture-only control pose
1 additional varied camera pose
1 light setting
CLIP ViT-B/16 final features only
surface_normal_aggregate only
```

Expected smoke outputs:

- render plan generated;
- Blender produces RGB/depth/normal/mask files;
- QC passes for most renders;
- labels build successfully;
- features extract successfully;
- a linear probe trains and saves metrics;
- an aggregate results CSV is produced.

Only after this works should Codex add DINOv2, intermediate layers, relative depth, and larger grids.

## Integration With Existing Repo

The existing repo already has dense depth/surface-normal training code and model wrappers. For this project:

- Reuse backbone wrappers if they provide clean frozen features.
- Do not force the MVP into the existing dense DPT probe setup.
- It is acceptable to create a separate `exp1/` package for manifest-driven rendering and cached-feature linear probes.
- Keep existing training scripts working.
- Avoid modifying global dataset builders unless necessary.

## Recommended Commands Eventually

```bash
python scripts/preprocess_assets.py --config configs/exp1_smoke.yaml
python scripts/create_render_plan.py --config configs/exp1_smoke.yaml
python scripts/export_render_chunks.py --config configs/exp1_smoke.yaml
blender --background --python scripts/render_blender.py -- --config configs/exp1_smoke.yaml --chunk data/manifests/render_chunks/chunk_000.jsonl
python scripts/validate_renders.py --config configs/exp1_smoke.yaml
python scripts/build_labels.py --config configs/exp1_smoke.yaml
python scripts/extract_exp1_features.py --config configs/exp1_smoke.yaml --model clip_vit_b16 --layers final
python scripts/train_exp1_probe.py --config configs/exp1_smoke.yaml --task surface_normal_aggregate --model clip_vit_b16 --layer final
python scripts/aggregate_exp1_results.py --config configs/exp1_smoke.yaml
python scripts/make_exp1_figures.py --config configs/exp1_smoke.yaml
```

## Definition of Done for MVP

The MVP is complete when:

- at least one smoke dataset renders successfully;
- object-disjoint splits are validated;
- all three texture conditions render and pass QC;
- texture-only control triplets have identical depth/normal buffers up to tolerance;
- CLIP and DINOv2 final features can be extracted;
- linear probes train for surface normals and relative depth;
- results are broken down by model and texture condition;
- generated outputs are reproducible from configs and manifests.

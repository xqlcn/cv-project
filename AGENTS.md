# AGENTS.md

## Project Goal

Build a controlled 3D-geometry probing experiment for visual foundation models, focused first on **Experiment 1: Textureless Geometry and Rendering Probes**.

The experiment should render the same ShapeNet/Objaverse objects in Blender under controlled conditions while holding mesh geometry fixed and varying texture, camera, lighting, and scale. The rendered dataset will be used to train **frozen-feature linear probes** on CLIP and DINOv2 representations to measure which geometry-related properties are encoded, how robust those encodings are to texture removal/noise, and whether geometric information appears more strongly in intermediate ViT layers than final representations.

The central research question is not simply whether CLIP “has geometry.” The goal is to identify:

- which geometry-related properties CLIP and DINOv2 encode;
- whether CLIP depends more heavily on photorealistic texture or semantic priors;
- whether DINOv2 is more robust for surface normals, relative depth, and viewpoint;
- whether early/intermediate CLIP layers contain more geometric information than final language-aligned layers;
- whether there are controlled settings where CLIP performs competitively with or better than DINOv2.

The minimum viable implementation should support:

- rendering objects under three texture conditions: `photorealistic`, `flat`, and `random_noise`;
- producing RGB, depth, normal, mask, and metadata outputs;
- training lightweight/frozen-feature probes for aggregate surface normal prediction and relative depth ordering;
- evaluating CLIP ViT-B/16, CLIP ViT-L/14, and DINOv2 ViT-B or ViT-L;
- probing final features and intermediate ViT layers such as 4, 8, and 12;
- reporting results by model, layer, task, and texture condition.

## Current Decisions

### Experiment scope

- Prioritize **Experiment 1** before chirality/Experiment 2.
- Treat Experiment 1 as a controlled synthetic-render benchmark rather than a natural-image benchmark.
- Use Blender to generate stimuli and ground-truth geometry buffers.
- Use ShapeNet and/or Objaverse objects as mesh sources.
- Mesh geometry must be fixed across texture conditions for the same object/render setting.
- Camera pose and lighting should be exactly matched across texture-only control triplets.
- The MVP can use a small object subset; the final run can scale to hundreds of objects if compute allows.

### Texture conditions

Implement exactly these texture conditions:

1. `photorealistic`
   - Preserve original or realistic imported material/texture when available.
   - If missing, use a documented neutral fallback and mark this in metadata.

2. `flat`
   - Override materials with a constant-color surface.
   - Keep lighting/shading active so geometry is still visible.
   - Do not use emission-only materials for the main flat condition.

3. `random_noise`
   - Apply deterministic random image/procedural texture uncorrelated with object category or geometry.
   - Seed must be stored in metadata.
   - Avoid category-specific colors or texture patterns.

### Tasks

MVP tasks:

- `surface_normal_aggregate`
  - Predict an aggregate foreground surface-normal vector from rendered RGB features.
  - Metric: angular error in degrees.

- `relative_depth_regions`
  - Predict which coarse image regions are closer to the camera.
  - Use a deterministic grid such as 3x3 regions and a fixed set of region pairs.
  - Metric: valid-pair accuracy and balanced accuracy.

Full tasks, after MVP works:

- `camera_distance`
  - Regression on camera-object distance, preferably log distance.

- `viewpoint`
  - Predict camera azimuth/elevation as either sin/cos regression or binned classification.

- `lighting_direction`
  - Predict dominant light direction as a unit vector or binned direction.

- `lighting_intensity`
  - Regression on raw or log light intensity.

- `apparent_scale`
  - Predict applied object scale and/or foreground mask area fraction.

### Models and layers

Evaluate at least:

- `clip_vit_b16`
- `clip_vit_l14`
- `dinov2_vit_b` or `dinov2_vit_l`

All backbones must remain frozen. Train only linear probes or very lightweight heads.

Layer probing:

- Probe final features.
- Probe intermediate layers, especially `layer4`, `layer8`, and `layer12` where available.
- For global MVP probes, use CLS/global features.
- Patch-token probing for dense/local normals/depth is optional and should be added only after the MVP pipeline is working.

### Splits and leakage prevention

- Splits must be **object-disjoint**, not render-disjoint.
- The same `object_id` must never appear in more than one of train/val/test.
- All joins must be by `render_id`; do not rely on row order.
- Store and validate split assignments in a manifest.

### Data and metadata

Use manifest-driven processing throughout the project. Every render should have a unique deterministic `render_id` and a manifest row containing:

- object/source/category fields;
- mesh path fields;
- split;
- texture condition and texture seed;
- camera distance, azimuth, elevation, FOV, pose/extrinsics;
- light direction/intensity;
- object scale;
- RGB/depth/normal/mask output paths;
- render status and QC fields.

## Tech Stack

### Existing repo stack

The attached repo is a reimplementation of `probe3d`, using:

- Python >= 3.9;
- PyTorch;
- Hydra/OmegaConf configs;
- loguru logging;
- open_clip_torch;
- transformers;
- albumentations;
- matplotlib/seaborn;
- pandas;
- pre-commit with black, isort, flake8, pyupgrade, and notebook output clearing.

The repo currently uses `setup.py develop` and package name `evals`.

### New Experiment 1 additions

Use:

- Blender + `bpy` for rendering;
- NumPy for geometry-buffer storage;
- Pandas + Parquet/CSV/JSONL for manifests;
- PyTorch for feature extraction and probes;
- open_clip_torch or existing `evals.models.clip` wrappers for CLIP;
- existing DINO/DINOv2 wrappers where possible;
- PyTest for unit tests;
- Matplotlib for plots;
- optionally Zarr/HDF5/memmap for large feature arrays.

Keep Blender-specific code isolated from model-training code. Blender often uses its own Python environment, so do not assume the project Conda environment is available inside Blender.

## Known Repository Structure

The attached archive currently contains roughly:

```text
probe3d-main/
  README.md
  requirements.txt
  setup.py
  .pre-commit-config.yaml
  configs/
    backbone/
      clip_b16.yaml
      clip_l14.yaml
      dinov2_b14.yaml
      dinov2_l14.yaml
      ...
    dataset/
      navi.yaml
      navi_reldepth.yaml
      nyu.yaml
    optimizer/
      ten_epoch.yaml
    probe/
      depth_dpt.yaml
      snorm_dpt.yaml
    depth_training.yaml
    snorm_training.yaml
    navi_correspondence.yaml
    scannet_correspondence.yaml
    spair_correspondence.yaml
  data_processing/
    README.md
    create_nyu_pkl.py
    resize_navi.py
    parse_spair_keypoints.py
  evals/
    datasets/
      builder.py
      navi.py
      nyu.py
      old_navi.py
      scannet_pairs.py
      spair.py
      utils.py
    models/
      clip.py
      dino.py
      probes.py
      convnext.py
      deit.py
      mae.py
      siglip.py
      ...
    utils/
      losses.py
      metrics.py
      optim.py
      transformations.py
      correlation.py
      correspondence.py
  train_depth.py
  train_snorm.py
  evaluate_navi_correspondence.py
  evaluate_scannet_correspondence.py
  evaluate_spair_correspondence.py
```

## Recommended New Structure for This Project

Prefer adding new Experiment 1 code without breaking the existing repo API:

```text
probe3d-main/
  AGENTS.md
  docs/
    codex/
      EXPERIMENT1_HANDOFF.md
      EXPERIMENT1_TASKS.md
  configs/
    exp1_smoke.yaml
    exp1_mvp.yaml
    exp1_full.yaml
    exp1/
      render.yaml
      paths.yaml
      tasks.yaml
  scripts/
    preprocess_assets.py
    create_render_plan.py
    export_render_chunks.py
    render_blender.py
    validate_renders.py
    make_contact_sheet.py
    build_labels.py
    extract_exp1_features.py
    train_exp1_probe.py
    train_all_exp1_probes.py
    aggregate_exp1_results.py
    make_exp1_figures.py
  exp1/
    __init__.py
    assets/
      discover.py
      normalize.py
      validate.py
    metadata/
      schema.py
      manifest.py
    rendering/
      grid.py
      camera.py
      lighting.py
      materials.py
      passes.py
      qc.py
    tasks/
      surface_normals.py
      relative_depth.py
      camera.py
      lighting.py
      scale.py
    data/
      image_dataset.py
      feature_dataset.py
      splits.py
      label_join.py
    features/
      extract.py
      storage.py
    probes/
      heads.py
      losses.py
      train.py
      metrics.py
      checkpoints.py
    evaluation/
      metrics.py
      comparisons.py
      tables.py
    analysis/
      plots.py
      reports.py
  tests/
    test_exp1_manifest.py
    test_exp1_render_grid.py
    test_exp1_labels.py
    test_exp1_splits.py
    test_exp1_probe_metrics.py
```

If you decide to place new code under `evals/`, use a clear namespace such as `evals/exp1/`. Avoid scattering Experiment 1 logic across unrelated existing model/dataset files.

## Coding Conventions

Follow the existing repository style unless there is a strong reason not to:

- Python 3.9-compatible syntax.
- Use Hydra/OmegaConf for experiment configs.
- Use black-compatible formatting with max line length 88.
- Use isort with black profile.
- Keep imports explicit and organized.
- Prefer small modules with clear responsibilities.
- Use deterministic seeds for data splitting, texture generation, and render-plan sampling.
- Add type hints for new public functions where practical.
- Use `pathlib.Path` for filesystem paths in normal Python code.
- Use explicit `render_id` joins instead of positional assumptions.
- Save configs and metadata with every experiment output.
- Use `torch.no_grad()` or `torch.inference_mode()` for frozen feature extraction.
- Set `model.eval()` for all frozen backbones.
- Do not apply random augmentations during feature extraction.
- Keep Blender scripts able to run from command line with `blender --background --python ... -- ...`.
- Keep Blender imports minimal; do not require the full ML environment inside Blender.
- Write tests for pure-Python components even if Blender rendering itself is harder to unit test.

## Key Files and Docs Codex Should Read

Read these first:

1. `README.md`
   - Existing environment setup and repo purpose.

2. `data_processing/README.md`
   - Existing dataset assumptions and preprocessing style.

3. `configs/depth_training.yaml` and `configs/snorm_training.yaml`
   - Existing Hydra training config style.

4. `configs/backbone/*.yaml`
   - Existing backbone naming and construction conventions.

5. `evals/models/clip.py`, `evals/models/dino.py`, and related model wrappers
   - Reuse existing CLIP/DINO feature-extraction wrappers if practical.

6. `evals/models/probes.py`
   - Existing dense probe heads. For Experiment 1 MVP, new heads should be simpler linear probes over cached features, but this file shows current probe style.

7. `evals/utils/metrics.py` and `evals/utils/losses.py`
   - Reuse metric/loss utilities where appropriate.

8. `train_depth.py` and `train_snorm.py`
   - Existing training-loop and Hydra/DDP conventions. Do not copy dense-probe assumptions blindly.

9. `Untitled document.pdf`
   - Main Experiment 1 and Experiment 2 description.

10. `cv Proposal.pdf`
   - Short project proposal and minimum viable Experiment 1 scope.

Extra handoff docs in this package:

- `docs/codex/EXPERIMENT1_HANDOFF.md`
- `docs/codex/EXPERIMENT1_TASKS.md`

## Constraints

- Do not include secrets, API keys, tokens, credentials, or private absolute paths.
- Do not commit raw datasets, rendered images, feature arrays, checkpoints, or large generated artifacts.
- All dataset paths must be configurable.
- The project should be runnable on a single GPU for the MVP.
- Backbones must remain frozen for the core experiment.
- Train only linear probes or explicitly documented lightweight heads.
- Use object-disjoint splits.
- Keep texture-only control renders exactly matched in camera, lighting, and scale.
- Store enough metadata to reproduce every render.
- Make every randomized choice seedable and reproducible.
- Do not silently drop failed renders; mark them in the manifest with an error reason.
- Do not let render failures crash an entire batch unless the config explicitly requests fail-fast behavior.
- Prefer Parquet for large tabular manifests when available, but support JSONL/CSV where Blender integration requires simpler dependencies.

## Open Tasks

### Phase 0: Repo preparation

- Add this `AGENTS.md` to the repo root.
- Add `docs/codex/EXPERIMENT1_HANDOFF.md` and `docs/codex/EXPERIMENT1_TASKS.md`.
- Decide whether Experiment 1 code lives under `exp1/` or `evals/exp1/`.
- Add smoke/MVP/full Hydra configs.

### Phase 1: Manifests and assets

- Implement asset discovery for ShapeNet/Objaverse meshes.
- Normalize meshes to a consistent centered unit scale.
- Generate object-disjoint train/val/test splits.
- Define manifest schema and deterministic `render_id` generation.
- Generate controlled render plans.

### Phase 2: Blender rendering

- Implement Blender renderer for RGB, depth, normals, masks, and metadata.
- Implement photorealistic/flat/random-noise material modes.
- Export render chunks to JSONL.
- Add render QC and contact-sheet generation.

### Phase 3: Labels

- Build aggregate surface-normal labels.
- Build relative-depth region-pair labels.
- Add camera, viewpoint, lighting, and apparent-size labels for the full version.

### Phase 4: Features

- Extract frozen CLIP and DINOv2 features.
- Save per-model/per-layer feature caches.
- Support final and intermediate layers.
- Ensure feature files include `render_id` arrays.

### Phase 5: Probes and evaluation

- Train linear probes for MVP tasks.
- Add cross-texture and within-texture train/test modes.
- Aggregate results by model/layer/task/texture.
- Add object-level bootstrap confidence intervals.
- Generate plots and summary tables.

### Phase 6: End-to-end reproducibility

- Add a smoke test pipeline.
- Add a Makefile or runner script.
- Add documentation for exact MVP commands.
- Ensure all generated outputs are ignored by Git.

## Things Codex Should Avoid

- Do not train or fine-tune CLIP/DINOv2 backbones for the core experiment.
- Do not use render-disjoint splits; this would leak object identity.
- Do not infer labels from filenames when manifest columns exist.
- Do not rely on row order when joining manifests, labels, features, or predictions.
- Do not add random image augmentations during feature extraction or probe training unless explicitly requested as an ablation.
- Do not replace the existing repo’s depth/surface-normal training scripts unless necessary.
- Do not assume ShapeNet/Objaverse paths are available on every machine.
- Do not hardcode local filesystem paths.
- Do not commit large generated outputs.
- Do not silently substitute missing photorealistic textures without recording fallback status.
- Do not make random-noise textures category-correlated.
- Do not make flat textures unlit/emissive unless implementing a separate ablation.
- Do not evaluate on failed or invalid-QC renders.
- Do not report a single aggregate score without breaking down by model, layer, task, and texture condition.
- Do not use dense DPT-style heads for the MVP unless explicitly changing the experiment; the core should use cached frozen features and linear probes.

## Suggested First Codex Task

Implement the manifest and render-plan foundation before touching Blender or model code:

1. Add `exp1/metadata/schema.py` and `exp1/metadata/manifest.py`.
2. Add deterministic `render_id` generation.
3. Add validation for required columns, split labels, texture conditions, and duplicate IDs.
4. Add `scripts/create_render_plan.py` using a small synthetic asset manifest.
5. Add tests for render-plan uniqueness, balanced texture conditions, and texture-only control matching.

This creates the spine that every later stage can consume.

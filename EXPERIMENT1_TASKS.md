# Experiment 1 Implementation Task List

This is a concrete task list Codex can execute incrementally. Each task should include tests or a smoke check before moving on.

## Task 1: Add Experiment 1 Configs

Create:

```text
configs/exp1_smoke.yaml
configs/exp1_mvp.yaml
configs/exp1_full.yaml
configs/exp1/paths.yaml
configs/exp1/render.yaml
configs/exp1/tasks.yaml
```

Required config groups:

- paths;
- experiment seed/name;
- asset sources;
- render resolution;
- texture conditions;
- camera grid;
- lighting grid;
- scale grid;
- split fractions;
- model/layer list;
- enabled tasks;
- probe hyperparameters.

Acceptance criteria:

- configs load with OmegaConf/Hydra;
- smoke config is tiny and safe to run;
- no absolute local paths are required.

## Task 2: Manifest Schema and Validation

Create:

```text
exp1/metadata/schema.py
exp1/metadata/manifest.py
tests/test_exp1_manifest.py
```

Implement:

- deterministic `render_id` generation;
- manifest save/load helpers;
- required-column validation;
- valid split validation;
- valid texture-condition validation;
- duplicate ID detection.

Acceptance criteria:

- tests pass;
- invalid manifests fail loudly with useful messages.

## Task 3: Synthetic Render Plan Generator

Before asset discovery exists, support generating a render plan from a tiny synthetic asset manifest. This allows early tests without real ShapeNet/Objaverse data.

Create:

```text
scripts/create_render_plan.py
exp1/rendering/grid.py
tests/test_exp1_render_grid.py
```

Acceptance criteria:

- every object appears in all three texture conditions;
- texture-only control rows match camera/lighting/scale exactly across textures;
- render IDs are unique;
- split labels are preserved.

## Task 4: Asset Discovery and Normalization

Create:

```text
scripts/preprocess_assets.py
exp1/assets/discover.py
exp1/assets/normalize.py
exp1/assets/validate.py
tests/test_exp1_assets.py
```

Support `.obj`, `.glb`, `.gltf`, `.fbx`, and `.ply` if practical. Export normalized meshes as `.glb`.

Acceptance criteria:

- asset manifest generated;
- normalized meshes are centered and scaled;
- invalid meshes are marked rather than crashing the whole run;
- object-disjoint split file is created.

## Task 5: Blender Renderer Skeleton

Create:

```text
scripts/render_blender.py
exp1/rendering/camera.py
exp1/rendering/lighting.py
exp1/rendering/materials.py
exp1/rendering/passes.py
```

Implement a single-record render first, then JSONL chunk rendering.

Acceptance criteria:

- renders one toy mesh;
- outputs `rgb.png`, `depth.npy`, `normal_*.npy`, `mask.npy`, `render_meta.json`;
- files have expected shapes;
- failure is logged per record.

## Task 6: Material Modes

Implement:

- original/photorealistic material preservation;
- flat constant-color material;
- deterministic random-noise texture generation.

Acceptance criteria:

- fixed camera/light texture triplet renders in all three modes;
- RGB differs across modes;
- depth/normal/mask match across texture-only controls up to tolerance;
- missing photorealistic textures are recorded as fallback.

## Task 7: Render QC

Create:

```text
scripts/validate_renders.py
scripts/make_contact_sheet.py
exp1/rendering/qc.py
```

QC checks:

- file existence;
- RGB/depth/normal/mask shape;
- finite foreground depth;
- normal norm;
- foreground mask fraction;
- texture-only geometry consistency.

Acceptance criteria:

- QC manifest written;
- contact sheets generated;
- invalid renders are excluded downstream.

## Task 8: Label Construction

Create:

```text
scripts/build_labels.py
exp1/tasks/surface_normals.py
exp1/tasks/relative_depth.py
exp1/tasks/camera.py
exp1/tasks/lighting.py
exp1/tasks/scale.py
tests/test_exp1_labels.py
```

Acceptance criteria:

- aggregate surface-normal labels generated;
- relative-depth labels generated with valid masks;
- unit tests cover synthetic normal/depth maps;
- labels join to render manifest by `render_id`.

## Task 9: Dataset and Split Utilities

Create:

```text
exp1/data/image_dataset.py
exp1/data/feature_dataset.py
exp1/data/splits.py
exp1/data/label_join.py
tests/test_exp1_splits.py
```

Acceptance criteria:

- object-disjoint split assertion works;
- datasets filter by split and texture condition;
- joins use `render_id`, never row order;
- relative-depth labels can be loaded as multi-output target + valid mask.

## Task 10: Feature Extraction

Create:

```text
scripts/extract_exp1_features.py
exp1/features/extract.py
exp1/features/storage.py
```

Reuse existing model wrappers if practical.

Acceptance criteria:

- frozen CLIP final features extract for smoke renders;
- feature cache includes `render_ids` and `features`;
- no random augmentation;
- features are finite and deterministic;
- intermediate layers can be added after final features work.

## Task 11: Linear Probe Training

Create:

```text
scripts/train_exp1_probe.py
scripts/train_all_exp1_probes.py
exp1/probes/heads.py
exp1/probes/losses.py
exp1/probes/train.py
exp1/probes/metrics.py
exp1/probes/checkpoints.py
tests/test_exp1_probe_metrics.py
```

Acceptance criteria:

- surface-normal probe trains and reports angular error;
- relative-depth probe trains and reports valid-pair accuracy;
- checkpoints/predictions/metrics saved;
- synthetic overfit tests pass;
- object split leakage check runs before training.

## Task 12: Aggregation and Figures

Create:

```text
scripts/aggregate_exp1_results.py
scripts/make_exp1_figures.py
exp1/evaluation/metrics.py
exp1/evaluation/comparisons.py
exp1/analysis/plots.py
```

Acceptance criteria:

- one long-format results table exists;
- results can be grouped by task/model/layer/texture;
- texture-dependence drops are computed;
- layer-wise plots are generated;
- object-level bootstrap confidence intervals are implemented if time permits.

## Task 13: Pipeline Runner

Create:

```text
scripts/run_exp1_pipeline.py
Makefile target(s) for exp1 smoke/MVP
```

Acceptance criteria:

- smoke pipeline can be run from documented commands;
- stages are idempotent when outputs already exist;
- Blender chunk shell script can be generated;
- generated outputs are ignored by Git.

## Do First

Start with Tasks 1-3. They require no Blender, no GPU, and no real dataset. They establish the stable manifest contract for every later component.

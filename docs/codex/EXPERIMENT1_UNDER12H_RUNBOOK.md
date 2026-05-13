# Experiment 1 Under-12-Hour Runbook

## Main Sampled Benchmark

- Objects: 480 total, 8 categories x 60 objects/category.
- Categories: airplane, bench, bottle, car, chair, lamp, sofa, table.
- Splits: category-stratified and object-disjoint, 42 train / 9 val / 9 test per category.
- Renders: 6 sampled pose/light/scale settings per object x 3 matched texture conditions = 8,640 rows.
- Main camera distance: fixed at 2.8.
- Tasks: aggregate normals, foreground-bbox relative depth, viewpoint, lighting direction, lighting intensity, apparent scale.
- Models: CLIP ViT-B/16, CLIP ViT-L/14, DINOv2 ViT-B at 224px.
- Layers: final, layer4, layer8, layer12.

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_under12h.yaml \
  --stages prepare

BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender \
RENDER_JOBS=4 \
bash data/exp1_under12h/manifests/render_chunks/run_blender_chunks.sh

# For multiple machines, distribute one line per worker from:
# data/exp1_under12h/manifests/render_chunks/render_chunk_commands.txt

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_under12h.yaml \
  --stages post_render
```

Run feature extraction and probes on Colab using:

- `notebooks/colab_exp1_under12h_global_features.ipynb`
- `notebooks/colab_exp1_under12h_patch_features.ipynb`
- `notebooks/colab_exp1_under12h_probes.ipynb`

## Camera-Distance Add-On

- Objects: 160 total, 8 categories x 20 objects/category.
- Renders: 2 sampled views x 3 distances x 3 textures = 2,880 rows.
- Config: `configs/exp1_under12h_camera_distance.yaml`.

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_under12h_camera_distance.yaml \
  --stages prepare

BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender \
RENDER_JOBS=4 \
bash data/exp1_under12h_camera_distance/manifests/render_chunks/run_blender_chunks.sh

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_under12h_camera_distance.yaml \
  --stages post_render
```

## Dense Patch-Depth Sub-Study

- Objects: 160 total, 8 categories x 20 objects/category.
- Renders: 3 sampled main settings x 3 textures = 1,440 rows.
- Models/layers: CLIP ViT-B/16 final/layer8 and DINOv2 ViT-B final/layer8.
- No CLIP-L/14 dense run by default.
- Within-texture only by default.
- Config: `configs/exp1_under12h_dense.yaml`.

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_under12h_dense.yaml \
  --stages prepare

BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender \
RENDER_JOBS=4 \
bash data/exp1_under12h_dense/manifests/render_chunks/run_blender_chunks.sh

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_under12h_dense.yaml \
  --stages post_render
```

## Runtime Budget

- Rendering main + camera add-on: expected 1.0-1.5h including chunk/QC overhead at the observed local rate.
- Label building and contact sheets: about 0.5h.
- Global feature extraction on one Colab T4: about 1.5-2h sequential.
- Dense patch features subset: about 1-1.5h.
- Probe training, aggregation, figures: about 1.5-2.5h.

Expected end-to-end wall time is 6-8h with a conservative ceiling around 10-11h, excluding asset downloads.

## ShapeNet Access

`ShapeNet/ShapeNetCore` is gated for file downloads. Before `prepare`, authenticate
without putting tokens in chat:

```bash
HF_HOME=/Users/jerry/cv-project/data/hf_cache \
  /Users/jerry/cv-project/.venv/bin/python -m huggingface_hub.commands.huggingface_cli login
```

The pipeline will also use `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` if one is set
in the shell that launches the run.

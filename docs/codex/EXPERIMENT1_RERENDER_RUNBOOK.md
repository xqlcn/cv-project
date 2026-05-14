# Experiment 1 Rerender Runbook

This runbook describes the canonical command sequence that reproduces the
two final analysis artifacts of Experiment 1:

- `outputs/exp1_rerender_analysis_figures/` — aggregate figures, summary
  CSVs, qualitative grids, and state-space plots.
- `outputs/exp1_dense_probe_analysis/` — the standalone dense-probe report
  with `analysis.md`, `metrics_summary.csv`, and supporting figures.

## Main Benchmark (`configs/exp1_main.yaml`)

- Objects: 480 total, 8 categories × 60 objects/category.
- Categories: airplane, bench, bottle, car, chair, lamp, sofa, table.
- Splits: category-stratified and object-disjoint, 42 train / 9 val / 9 test
  per category.
- Renders: 6 sampled pose/light/scale settings per object × 3 matched
  texture conditions = 8,640 rows.
- Main camera distance: fixed at 2.8.
- Tasks: aggregate normals, foreground-bbox relative depth, viewpoint,
  lighting direction, lighting intensity, apparent scale.
- Models: CLIP ViT-B/16, CLIP ViT-L/14, DINOv2 ViT-B at 224px.
- Layers: final, layer4, layer8, layer12.

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_main.yaml \
  --stages prepare

BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender \
RENDER_JOBS=4 \
bash data/exp1_main/manifests/render_chunks/run_blender_chunks.sh

# For multiple machines, distribute one line per worker from:
# data/exp1_main/manifests/render_chunks/render_chunk_commands.txt

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_main.yaml \
  --stages post_render
```

Run frozen feature extraction and probe training on Colab using:

- `notebooks/colab_exp1_zip_feature_extraction.ipynb`
- `notebooks/colab_exp1_zip_probe_training.ipynb`
- `notebooks/colab_exp1_zip_probe_training_wandb.ipynb` (dense, W&B logged)

## Dense Patch-Depth Sub-Study (`configs/exp1_dense.yaml`)

- Objects: 160 total, 8 categories × 20 objects/category.
- Renders: 3 sampled main settings × 3 textures = 1,440 rows.
- Models/layers: CLIP ViT-B/16 final/layer8 and DINOv2 ViT-B final/layer8.
- No CLIP-L/14 dense run by default.
- Within-texture only by default.

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_dense.yaml \
  --stages prepare

BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender \
RENDER_JOBS=4 \
bash data/exp1_dense/manifests/render_chunks/run_blender_chunks.sh

PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_dense.yaml \
  --stages post_render
```

## Final Analysis Stage

After both probe trees exist under `outputs/exp1_main/` and
`outputs/exp1_dense/`, build the user-facing deliverables:

```bash
PYTHONPATH=. python scripts/analysis/plot_exp1_rerender_summary.py \
  --main-dir  outputs/exp1_main \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analysis/plot_exp1_dense_reliability.py \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analysis/plot_exp1_state_space_breakdown.py \
  --main-dir  outputs/exp1_main \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analysis/plot_exp1_qualitative_examples.py \
  --dense-dir outputs/exp1_dense

PYTHONPATH=. python scripts/analyze_exp1_dense_probes.py
```

## Runtime Budget

- Rendering main: expected 1.0–1.5 h including chunk/QC overhead at the
  observed local rate.
- Label building and contact sheets: about 0.5 h.
- Global feature extraction on one Colab T4: about 1.5–2 h sequential.
- Dense patch features subset: about 1–1.5 h.
- Probe training, aggregation, figures: about 1.5–2.5 h.

Expected end-to-end wall time is 6–8 h with a conservative ceiling around
10–11 h, excluding asset downloads.

## ShapeNet Access

`ShapeNet/ShapeNetCore` is gated for file downloads. Before `prepare`,
authenticate without putting tokens in chat:

```bash
HF_HOME=$PWD/data/hf_cache \
  .venv/bin/python -m huggingface_hub.commands.huggingface_cli login
```

The pipeline will also use `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` if one is
set in the shell that launches the run.

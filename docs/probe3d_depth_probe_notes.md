# Probe3D Depth Probe Notes

## Probe3D Code Map

- `train_depth.py`: Hydra entrypoint for dense depth probe training and final scale-aware / scale-invariant evaluation.
- `evals/models/probes.py`: `DepthHead`, `Linear`, `MultiscaleHead`, and `DPT` probe heads.
- `evals/utils/losses.py`: `DepthLoss`, `sig_loss`, and log-depth gradient loss.
- `evals/utils/metrics.py`: `evaluate_depth`, `depth_rmse`, and least-squares `match_scale_and_shift`.
- `evals/datasets/nyu.py` and `evals/datasets/navi.py`: RGB-D loading, invalid depth handling, shared image/target transforms, and depth/normal labels.
- `evals/models/clip.py`, `evals/models/dino.py`, and related wrappers: dense, CLS, GAP, dense-CLS, final-layer, and intermediate-layer feature extraction.

## Probe3D Depth Training

Probe3D defaults to `configs/probe/depth_dpt.yaml`, which instantiates `DepthHead` with `head_type: dpt`, `prediction_type: bindepth`, `hidden_dim: 512`, and `kernel_size: 3`. This is a lightweight DPT-style dense head, not a purely linear probe.

`DepthHead` also supports a `linear` mode: one convolution over dense features, equivalent to a per-location linear map after feature upsampling. That is the design we adopt for Experiment 1 because the scientific question is about linearly decodable geometry in frozen CLIP/DINOv2 features.

Probe3D uses `AdamW`, `probe_lr=5e-4`, `model_lr=0`, `n_epochs=10`, and linear warmup plus cosine decay. With `model_lr=0`, the backbone is run under `torch.no_grad()` and detached.

The depth loss is `10 * sig_loss + 0.5 * gradient_loss`, both in log depth and both ignoring invalid target pixels. We borrow the invalid-depth masking and scale-aware / scale-invariant evaluation, but keep our primary training loss as masked SSI L1 over patch-pooled render depths.

## Representations Probed

Probe3D wrappers can emit:

- `cls`: CLS/global token only.
- `gap`: mean patch token.
- `dense`: patch-token feature map.
- `dense-cls`: patch-token map concatenated with broadcast CLS.
- Four intermediate layers when `return_multilayer=True`, roughly 1/4, 1/2, 3/4, and final transformer depth.

Experiment 1 now mirrors the useful pieces with cached patch grids and optional cached CLS features. The default dense-depth result uses `feature_mode: patch`; `feature_mode: patch_cls` is an ablation.

## Probe3D Data And Labels

Probe3D uses NYUv2 and NAVI for depth/surface-normal probing. These are natural RGB-D or object-centric multiview datasets, not texture-controlled synthetic renders. Their labels are dense metric or normalized depth maps with invalid pixels encoded as zero. Image and target transforms are shared so depth maps remain aligned with model inputs.

Experiment 1 keeps Blender-rendered ShapeNet/Objaverse RGB, depth, normal, and mask outputs. Depth remains positive camera-space z distance with invalid background masked out.

## Adopted Ideas

- Dense patch-aligned depth targets instead of only global CLS labels.
- Invalid depth/mask handling at the loss and metric level.
- Scale-aware and scale-invariant depth metrics.
- Optional patch+CLS feature mode as a Probe3D-style dense-CLS ablation.
- Intermediate-layer probing with cached patch grids.
- Linear warmup plus cosine learning-rate decay for dense probes.

## Rejected Ideas

- Replacing controlled ShapeNet/Objaverse renders with NYU/NAVI.
- Random image augmentations during feature extraction or probe training.
- Online backbone execution during probe training.
- DPT as the default head. DPT-style fusion can be added later only as a capacity ablation.
- Training or fine-tuning CLIP/DINOv2 backbones.

## Proposed Experiment 1 Pipeline Changes

The dense depth task is named `dense_depth_patches`. It consumes patch feature caches with:

- `render_ids`
- `patch_features`: `[N, P, P, D]`
- optional `cls_features`: `[N, D_cls]`
- `patch_grid_shape`
- `feature_dim`
- metadata including `model_input_size`, `patch_size`, `feature_mode`, `model_name`, and `layer_name`

Targets are built lazily from manifest rows by loading `depth.npy` and `mask.npy`, applying resize-shortest-edge plus center-crop alignment when the cache records a model input size, and pooling valid foreground depth to the exact patch grid. The default target statistic is median depth and the default valid threshold is `min_valid_fraction_per_patch: 0.25`.

## Commands

Prepare the dense sub-study:

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_dense.yaml \
  --stages prepare
```

Render with the generated Blender chunk script, then post-process:

```bash
PYTHONPATH=. python scripts/run_exp1_pipeline.py \
  --config configs/exp1_dense.yaml \
  --stages post_render
```

Extract patch grids with CLS features:

```bash
PYTHONPATH=. python scripts/extract_exp1_patch_features.py \
  --config configs/exp1_dense.yaml \
  --models clip_vit_b16 dinov2_vit_b \
  --layers final layer4 layer8 layer12 \
  --include-cls
```

Train and aggregate dense probes:

```bash
PYTHONPATH=. python scripts/train_all_dense_depth_probes.py \
  --config configs/exp1_dense.yaml

PYTHONPATH=. python scripts/aggregate_exp1_results.py \
  --config configs/exp1_dense.yaml
```

## Verification Checklist

- RGB, depth, and mask shapes match for selected render rows.
- Foreground depth is finite and positive.
- Patch cache row count, render IDs, grid shape, and feature dimension are valid.
- Optional `cls_features` are present before using `feature_mode: patch_cls`.
- Pooled dense targets have the same `[P, P]` grid as patch features.
- A single CPU forward/backward pass through `DenseDepthHead` succeeds.
- Metrics include SSI L1, scale-aware depth metrics, scale-invariant depth metrics, Pearson correlation, and valid patch counts.
- Aggregation includes dense-depth metrics and object-level bootstrap CIs from `predictions.csv`.


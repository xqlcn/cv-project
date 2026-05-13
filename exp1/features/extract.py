"""Frozen feature extraction for Experiment 1 render manifests."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from exp1.data.splits import filter_manifest
from exp1.features.storage import (
    patch_feature_cache_path,
    save_feature_cache,
    save_patch_feature_cache,
)
from exp1.metadata.manifest import load_manifest
from src.models.clip_extractor import CLIPExtractorConfig, FrozenCLIPExtractor
from src.models.dino_extractor import DINOExtractorConfig, FrozenDINOv2Extractor

BatchFn = Callable[[list[Image.Image]], Mapping[str, Any]]
CLS_FEATURE_SUFFIX = "__cls_features"


def manifest_rows_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash render-row identity fields used to guard feature-cache freshness."""
    payload = []
    for row in rows:
        payload.append(
            {
                "render_id": str(row.get("render_id", "")),
                "rgb_path": str(row.get("rgb_path", "")),
                "texture_condition": str(row.get("texture_condition", "")),
                "split": str(row.get("split", "")),
                "object_id": str(row.get("object_id", "")),
            }
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def parse_layer_name(layer_name: str) -> Optional[int]:
    """Convert 'final' or 'layerN' to the integer layer used by wrappers."""
    text = str(layer_name).strip().lower()
    if text == "final":
        return None
    if text.startswith("layer"):
        suffix = text.removeprefix("layer")
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"Unsupported layer name: {layer_name}")


def requested_layer_numbers(layer_names: Sequence[str]) -> list[int]:
    """Return non-final layer numbers requested by layer names."""
    layers = []
    for layer_name in layer_names:
        parsed = parse_layer_name(str(layer_name))
        if parsed is not None:
            layers.append(parsed)
    return sorted(set(layers))


def _resolve_path(project_root: Optional[Union[str, Path]], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


def load_render_rows_for_features(
    manifest_path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
    split: Optional[Union[str, Sequence[str]]] = None,
    texture_condition: Optional[Union[str, Sequence[str]]] = None,
    require_qc_pass: bool = True,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load render rows and apply deterministic feature-extraction filters."""
    df = load_manifest(manifest_path, validate=False)
    df = filter_manifest(
        df,
        split=split,
        texture_condition=texture_condition,
        require_qc_pass=require_qc_pass,
    )
    if limit is not None:
        df = df.iloc[: int(limit)].reset_index(drop=True)
    if "render_id" not in df.columns or "rgb_path" not in df.columns:
        raise ValueError("Feature extraction requires render_id and rgb_path columns")
    for value in df["rgb_path"].tolist():
        path = _resolve_path(project_root, value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing RGB image for feature extraction: {path}")
    return df.reset_index(drop=True)


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def _batch_layer_features(
    batch_features: Mapping[str, Any],
    *,
    layer_name: str,
    token: str,
) -> np.ndarray:
    layer_number = parse_layer_name(layer_name)
    if layer_number is None:
        if token == "cls":
            return _tensor_to_numpy(batch_features["cls_final"])
        if "patch_mean" in batch_features:
            return _tensor_to_numpy(batch_features["patch_mean"])
        patch = batch_features["patch_tokens_final"]
        if isinstance(patch, torch.Tensor):
            return _tensor_to_numpy(patch.mean(dim=1))
        return np.asarray(patch, dtype=np.float32).mean(axis=1)

    layer_map = batch_features.get("layer_cls", {})
    if layer_number not in layer_map:
        raise KeyError(f"Extractor output is missing layer {layer_number}")
    return _tensor_to_numpy(layer_map[layer_number])


def _normalize_features(features: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    return F.normalize(tensor, dim=1).numpy()


def extract_feature_arrays(
    rows: Iterable[Mapping[str, Any]],
    *,
    batch_extract_fn: BatchFn,
    layer_names: Sequence[str],
    batch_size: int,
    token: str = "cls",
    project_root: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    image_column: str = "rgb_path",
    show_progress: bool = True,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """Extract feature arrays for requested layers using a batch callback."""
    rows_list = [dict(row) for row in rows]
    layer_names = [str(layer) for layer in layer_names]
    features_by_layer: dict[str, list[np.ndarray]] = {
        layer: [] for layer in layer_names
    }
    iterator = range(0, len(rows_list), int(batch_size))
    if show_progress:
        iterator = tqdm(iterator, desc="extract exp1 features")

    def _load_image(row: Mapping[str, Any]) -> Image.Image:
        with Image.open(_resolve_path(project_root, row[image_column])) as image:
            return image.convert("RGB")

    worker_count = max(0, int(num_workers))
    for start in iterator:
        batch_rows = rows_list[start : start + int(batch_size)]
        if worker_count > 1 and len(batch_rows) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                images = list(executor.map(_load_image, batch_rows))
        else:
            images = [_load_image(row) for row in batch_rows]
        batch_features = batch_extract_fn(images)
        for layer_name in layer_names:
            layer_features = _batch_layer_features(
                batch_features,
                layer_name=layer_name,
                token=token,
            )
            if layer_features.shape[0] != len(batch_rows):
                raise ValueError(
                    f"Layer {layer_name} produced {layer_features.shape[0]} rows "
                    f"for batch size {len(batch_rows)}"
                )
            features_by_layer[layer_name].append(layer_features.astype(np.float32))

    out: dict[str, np.ndarray] = {}
    for layer_name, chunks in features_by_layer.items():
        arr = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0))
        if normalize and arr.size:
            arr = _normalize_features(arr)
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite features produced for layer {layer_name}")
        out[layer_name] = arr.astype(np.float32)
    return out


def build_backbone_extractor(
    *,
    model_name: str,
    model_cfg: Mapping[str, Any],
    layer_names: Sequence[str],
    device: torch.device,
    return_patch_layers: bool = False,
):
    """Instantiate a frozen CLIP or DINOv2 extractor from config."""
    family = str(model_cfg["family"])
    layers = requested_layer_numbers(layer_names)
    if family == "clip":
        return FrozenCLIPExtractor(
            CLIPExtractorConfig(
                model_id=str(model_cfg.get("model_id", "openai/clip-vit-base-patch16")),
                image_size=int(model_cfg.get("image_size", 224)),
                use_open_clip=bool(model_cfg.get("use_open_clip", False)),
                open_clip_model=str(model_cfg.get("open_clip_model", "ViT-B-16")),
                open_clip_pretrained=str(
                    model_cfg.get("open_clip_pretrained", "laion2b_s34b_b88k")
                ),
                layers=layers or None,
                return_patch_layers=bool(return_patch_layers),
            ),
            device=device,
        )
    if family == "dino":
        return FrozenDINOv2Extractor(
            DINOExtractorConfig(
                hub_name=str(model_cfg.get("hub_name", "dinov2_vitb14")),
                model_id=model_cfg.get("model_id"),
                backend=str(model_cfg.get("backend", "transformers")),
                revision=model_cfg.get("revision"),
                image_size=int(model_cfg.get("image_size", 518)),
                layers=layers or None,
                return_patch_layers=bool(return_patch_layers),
            ),
            device=device,
        )
    raise ValueError(f"Unknown model family for {model_name}: {family}")


def batch_extract_fn_for_extractor(
    extractor: Any,
    *,
    num_workers: int = 0,
    use_amp: bool = False,
) -> BatchFn:
    """Return a deterministic PIL batch callback for an existing extractor."""

    def _extract(images: list[Image.Image]) -> Mapping[str, Any]:
        device = getattr(extractor, "device", torch.device("cpu"))
        device_type = str(getattr(device, "type", device))
        amp_enabled = bool(use_amp) and device_type == "cuda"
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        if hasattr(extractor, "_pixel_values_from_pil"):
            pixel_values = extractor._pixel_values_from_pil(images)
            with torch.inference_mode(), amp_context:
                return extractor.forward_image_tensor(pixel_values)
        if hasattr(extractor, "_prepare_batch"):
            if int(num_workers) > 1 and hasattr(extractor, "transform"):
                with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
                    tensors = list(
                        executor.map(
                            lambda image: extractor.transform(image.convert("RGB")),
                            images,
                        )
                    )
                batch = torch.stack(tensors).to(extractor.device, non_blocking=True)
                with torch.inference_mode(), amp_context:
                    return extractor.forward_tensor(batch)
            batch = extractor._prepare_batch(images)
            with torch.inference_mode(), amp_context:
                return extractor.forward_tensor(batch)
        raise TypeError(f"Unsupported extractor type: {type(extractor).__name__}")

    return _extract


def _mapping_from_omegaconf(value: Any) -> Mapping[str, Any]:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    return value


def _batch_patch_features(
    batch_features: Mapping[str, Any],
    *,
    layer_name: str,
) -> np.ndarray:
    """Return per-layer patch tokens as a 3D array (B, num_patches, D)."""
    layer_number = parse_layer_name(layer_name)
    if layer_number is None:
        patch = batch_features["patch_tokens_final"]
    else:
        patch_map = batch_features.get("layer_patch", {})
        if layer_number not in patch_map:
            raise KeyError(
                f"Extractor output is missing patch tokens for layer {layer_number}"
            )
        patch = patch_map[layer_number]
    return _tensor_to_numpy(patch)


def _patch_cls_key(layer_name: str) -> str:
    return f"{str(layer_name)}{CLS_FEATURE_SUFFIX}"


def _batch_cls_features(
    batch_features: Mapping[str, Any],
    *,
    layer_name: str,
) -> np.ndarray:
    """Return the CLS/global token corresponding to a patch-token layer."""
    layer_number = parse_layer_name(layer_name)
    if layer_number is None:
        return _tensor_to_numpy(batch_features["cls_final"])
    layer_map = batch_features.get("layer_cls", {})
    if layer_number not in layer_map:
        raise KeyError(
            f"Extractor output is missing CLS features for layer {layer_number}"
        )
    return _tensor_to_numpy(layer_map[layer_number])


def _patch_grid_side(num_patches: int) -> int:
    side = int(round(np.sqrt(num_patches)))
    if side * side != num_patches:
        raise ValueError(
            f"Patch grid is not square: {num_patches} patches do not factor into PxP"
        )
    return side


def extract_patch_feature_arrays(
    rows: Iterable[Mapping[str, Any]],
    *,
    batch_extract_fn: BatchFn,
    layer_names: Sequence[str],
    batch_size: int,
    project_root: Optional[Union[str, Path]] = None,
    image_column: str = "rgb_path",
    show_progress: bool = True,
    num_workers: int = 0,
    dtype: str = "float16",
    include_cls: bool = True,
) -> dict[str, np.ndarray]:
    """Extract per-layer patch grids (N, P, P, D) for the given rows.

    When ``include_cls`` is true, the returned dictionary also contains
    ``f"{layer}__cls_features"`` arrays with shape ``(N, D_cls)``. The patch
    arrays remain available under the original layer names for backward
    compatibility with older callers.
    """
    rows_list = [dict(row) for row in rows]
    layer_names = [str(layer) for layer in layer_names]
    features_by_layer: dict[str, list[np.ndarray]] = {
        layer: [] for layer in layer_names
    }
    cls_by_layer: dict[str, list[np.ndarray]] = {layer: [] for layer in layer_names}
    iterator = range(0, len(rows_list), int(batch_size))
    if show_progress:
        iterator = tqdm(iterator, desc="extract exp1 patch features")

    def _load_image(row: Mapping[str, Any]) -> Image.Image:
        with Image.open(_resolve_path(project_root, row[image_column])) as image:
            return image.convert("RGB")

    worker_count = max(0, int(num_workers))
    grid_side: dict[str, int] = {}
    feat_dim: dict[str, int] = {}
    target_dtype = np.dtype(dtype)
    for start in iterator:
        batch_rows = rows_list[start : start + int(batch_size)]
        if worker_count > 1 and len(batch_rows) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                images = list(executor.map(_load_image, batch_rows))
        else:
            images = [_load_image(row) for row in batch_rows]
        batch_features = batch_extract_fn(images)
        for layer_name in layer_names:
            tokens = _batch_patch_features(batch_features, layer_name=layer_name)
            if tokens.shape[0] != len(batch_rows):
                raise ValueError(
                    f"Layer {layer_name} produced {tokens.shape[0]} rows "
                    f"for batch size {len(batch_rows)}"
                )
            side = _patch_grid_side(int(tokens.shape[1]))
            grid_side.setdefault(layer_name, side)
            feat_dim.setdefault(layer_name, int(tokens.shape[2]))
            grid = tokens.reshape(
                tokens.shape[0], side, side, tokens.shape[2]
            ).astype(target_dtype, copy=False)
            features_by_layer[layer_name].append(grid)
            if include_cls:
                cls = _batch_cls_features(batch_features, layer_name=layer_name)
                if cls.shape[0] != len(batch_rows):
                    raise ValueError(
                        f"Layer {layer_name} produced {cls.shape[0]} CLS rows "
                        f"for batch size {len(batch_rows)}"
                    )
                cls_by_layer[layer_name].append(cls.astype(target_dtype, copy=False))

    out: dict[str, np.ndarray] = {}
    for layer_name, chunks in features_by_layer.items():
        if not chunks:
            out[layer_name] = np.zeros((0, 0, 0, 0), dtype=target_dtype)
            if include_cls:
                out[_patch_cls_key(layer_name)] = np.zeros((0, 0), dtype=target_dtype)
            continue
        arr = np.concatenate(chunks, axis=0)
        out[layer_name] = arr
        if include_cls:
            out[_patch_cls_key(layer_name)] = np.concatenate(
                cls_by_layer[layer_name],
                axis=0,
            )
    return out


def extract_and_save_patch_caches(
    rows: Iterable[Mapping[str, Any]],
    *,
    model_name: str,
    model_cfg: Mapping[str, Any],
    layer_names: Sequence[str],
    feature_dir: Union[str, Path],
    batch_size: int,
    device: torch.device,
    project_root: Optional[Union[str, Path]] = None,
    num_workers: int = 0,
    dtype: str = "float16",
    use_amp: bool = False,
    include_cls: bool = True,
) -> list[Path]:
    """Extract one model's patch tokens for requested layers and save NPZ caches.

    For each requested layer we run the backbone once and write the cache before
    moving on to the next layer. That keeps peak memory bounded to one layer's
    cache (~5-10 GB at float16 for the bounded experiment) instead of holding
    every layer simultaneously.
    """
    rows_list = [dict(row) for row in rows]
    render_ids = [str(row["render_id"]) for row in rows_list]
    paths: list[Path] = []
    for layer_name in layer_names:
        extractor = build_backbone_extractor(
            model_name=model_name,
            model_cfg=_mapping_from_omegaconf(model_cfg),
            layer_names=[layer_name],
            device=device,
            return_patch_layers=True,
        )
        arrays = extract_patch_feature_arrays(
            rows_list,
            batch_extract_fn=batch_extract_fn_for_extractor(
                extractor,
                num_workers=num_workers,
                use_amp=use_amp,
            ),
            layer_names=[layer_name],
            batch_size=batch_size,
            project_root=project_root,
            num_workers=num_workers,
            dtype=dtype,
            include_cls=include_cls,
        )
        features = arrays[layer_name]
        cls_features = arrays.get(_patch_cls_key(layer_name)) if include_cls else None
        path = patch_feature_cache_path(
            feature_dir, model_name=model_name, layer_name=layer_name
        )
        input_size = model_cfg.get("image_size")
        patch_size = None
        if input_size is not None and features.size:
            patch_size = int(input_size) // int(features.shape[1])
        paths.append(
            save_patch_feature_cache(
                path,
                render_ids=render_ids,
                patch_features=features,
                cls_features=cls_features,
                metadata={
                    "model_name": model_name,
                    "layer_name": layer_name,
                    "feature_type": "patch_grid",
                    "manifest_hash": manifest_rows_fingerprint(rows_list),
                    "num_render_rows": len(rows_list),
                    "use_amp": bool(use_amp),
                    "dtype": dtype,
                    "feature_mode": "patch_cls" if include_cls else "patch",
                    "model_input_size": int(input_size) if input_size else None,
                    "patch_size": patch_size,
                    "preprocess_signature": (
                        f"resize_shortest_edge_center_crop_{int(input_size)}"
                        if input_size
                        else "unknown"
                    ),
                    "patch_grid_side": int(features.shape[1])
                    if features.size
                    else 0,
                    "feature_dim": int(features.shape[3]) if features.size else 0,
                    "cls_feature_dim": int(cls_features.shape[1])
                    if cls_features is not None and cls_features.size
                    else 0,
                },
                dtype=dtype,
            )
        )
        del features, arrays, extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return paths


def extract_and_save_feature_caches(
    rows: Iterable[Mapping[str, Any]],
    *,
    model_name: str,
    model_cfg: Mapping[str, Any],
    layer_names: Sequence[str],
    feature_dir: Union[str, Path],
    batch_size: int,
    token: str,
    device: torch.device,
    project_root: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    num_workers: int = 0,
    use_amp: bool = False,
) -> list[Path]:
    """Extract one model's requested layers and save canonical NPZ caches."""
    rows_list = [dict(row) for row in rows]
    extractor = build_backbone_extractor(
        model_name=model_name,
        model_cfg=_mapping_from_omegaconf(model_cfg),
        layer_names=layer_names,
        device=device,
    )
    arrays = extract_feature_arrays(
        rows_list,
        batch_extract_fn=batch_extract_fn_for_extractor(
            extractor,
            num_workers=num_workers,
            use_amp=use_amp,
        ),
        layer_names=layer_names,
        batch_size=batch_size,
        token=token,
        project_root=project_root,
        normalize=normalize,
        num_workers=num_workers,
    )
    render_ids = [str(row["render_id"]) for row in rows_list]
    paths: list[Path] = []
    for layer_name, features in arrays.items():
        path = Path(feature_dir) / str(model_name) / f"{layer_name}.npz"
        paths.append(
            save_feature_cache(
                path,
                render_ids=render_ids,
                features=features,
                metadata={
                    "model_name": model_name,
                    "layer_name": layer_name,
                    "token": token,
                    "feature_type": "global",
                    "manifest_hash": manifest_rows_fingerprint(rows_list),
                    "num_render_rows": len(rows_list),
                    "use_amp": bool(use_amp),
                    "normalized": bool(normalize),
                },
            )
        )
    return paths

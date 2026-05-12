"""Frozen feature extraction for Experiment 1 render manifests."""

from __future__ import annotations

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
from exp1.features.storage import save_feature_cache
from exp1.metadata.manifest import load_manifest
from src.models.clip_extractor import CLIPExtractorConfig, FrozenCLIPExtractor
from src.models.dino_extractor import DINOExtractorConfig, FrozenDINOv2Extractor

BatchFn = Callable[[list[Image.Image]], Mapping[str, Any]]


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

    for start in iterator:
        batch_rows = rows_list[start : start + int(batch_size)]
        images = []
        for row in batch_rows:
            with Image.open(_resolve_path(project_root, row[image_column])) as image:
                images.append(image.convert("RGB"))
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
            ),
            device=device,
        )
    if family == "dino":
        return FrozenDINOv2Extractor(
            DINOExtractorConfig(
                hub_name=str(model_cfg.get("hub_name", "dinov2_vitb14")),
                image_size=int(model_cfg.get("image_size", 518)),
                layers=layers or None,
            ),
            device=device,
        )
    raise ValueError(f"Unknown model family for {model_name}: {family}")


def batch_extract_fn_for_extractor(extractor: Any) -> BatchFn:
    """Return a deterministic PIL batch callback for an existing extractor."""

    def _extract(images: list[Image.Image]) -> Mapping[str, Any]:
        if hasattr(extractor, "_pixel_values_from_pil"):
            pixel_values = extractor._pixel_values_from_pil(images)
            return extractor.forward_image_tensor(pixel_values)
        if hasattr(extractor, "_prepare_batch"):
            batch = extractor._prepare_batch(images)
            return extractor.forward_tensor(batch)
        raise TypeError(f"Unsupported extractor type: {type(extractor).__name__}")

    return _extract


def _mapping_from_omegaconf(value: Any) -> Mapping[str, Any]:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    return value


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
        batch_extract_fn=batch_extract_fn_for_extractor(extractor),
        layer_names=layer_names,
        batch_size=batch_size,
        token=token,
        project_root=project_root,
        normalize=normalize,
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
                    "normalized": bool(normalize),
                },
            )
        )
    return paths

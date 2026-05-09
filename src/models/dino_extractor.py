"""Frozen DINOv2 feature extraction via torch.hub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


TokenKind = Literal["cls", "patch_mean"]


@dataclass
class DINOExtractorConfig:
    hub_name: str = "dinov2_vitb14"  # dinov2_vitb14 | dinov2_vitl14 | dinov2_vits14
    image_size: int = 518
    layers: Optional[List[int]] = None  # 1-based block indices (converted to 0-based internally)


class FrozenDINOv2Extractor(nn.Module):
    def __init__(self, cfg: DINOExtractorConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", cfg.hub_name)
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(cfg.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _prepare_batch(self, images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images])
        return batch.to(self.device, non_blocking=True)

    @staticmethod
    def _to_layer_indices_0based(layers_1based: List[int]) -> List[int]:
        return [max(0, i - 1) for i in layers_1based]

    def _forward_final_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks:
            x = blk(x)
        return self.model.norm(x)

    @torch.inference_mode()
    def forward_tensor(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        layer_cls: Dict[int, torch.Tensor] = {}
        if self.cfg.layers:
            idxs = self._to_layer_indices_0based(self.cfg.layers)
            blocks = self.model.get_intermediate_layers(
                x,
                n=idxs,
                reshape=False,
                return_class_token=True,
                norm=True,
            )
            for one_based, block_out in zip(self.cfg.layers, blocks):
                cls_tok = block_out[1]
                layer_cls[int(one_based)] = cls_tok

        tokens = self._forward_final_tokens(x)
        cls = tokens[:, 0, :]
        patch = tokens[:, 1:, :]
        out["cls_final"] = F.normalize(cls, dim=-1)
        out["patch_tokens_final"] = patch
        if layer_cls:
            out["layer_cls"] = layer_cls
        return out

    @torch.inference_mode()
    def extract_paths(
        self,
        image_paths: List[str],
        *,
        batch_size: int,
        token: TokenKind = "cls",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        from pathlib import Path

        from tqdm import tqdm

        results: Dict[str, Dict[str, torch.Tensor]] = {}
        for start in tqdm(range(0, len(image_paths), batch_size), desc="DINO extract"):
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(Path(p)).convert("RGB") for p in batch_paths]
            x = self._prepare_batch(images)
            feats = self.forward_tensor(x)
            for i, p in enumerate(batch_paths):
                entry: Dict[str, torch.Tensor] = {}
                if token == "cls":
                    entry["cls_final"] = feats["cls_final"][i].detach().cpu()
                else:
                    patch = feats["patch_tokens_final"][i]
                    entry["patch_mean"] = patch.mean(dim=0).detach().cpu()
                if "layer_cls" in feats:
                    entry["layer_cls"] = {k: v[i].detach().cpu() for k, v in feats["layer_cls"].items()}
                results[p] = entry
        return results

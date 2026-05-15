"""Frozen CLIP ViT feature extraction (HuggingFace or OpenCLIP)."""

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
class CLIPExtractorConfig:
    model_id: str = "openai/clip-vit-base-patch16"
    image_size: int = 224
    use_open_clip: bool = False
    open_clip_model: str = "ViT-B-16"
    open_clip_pretrained: str = "laion2b_s34b_b88k"
    layers: Optional[List[int]] = (
        None  # 1-based hidden_states index (HF); ignored for OpenCLIP
    )
    return_patch_layers: bool = False


class FrozenCLIPExtractor(nn.Module):
    """Frozen CLIP vision tower with optional multi-layer hidden states (HF only)."""

    def __init__(self, cfg: CLIPExtractorConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.processor = None
        self.hf_model = None
        self.open_clip_model = None
        self._open_clip_preprocess = None
        self.use_open_clip = cfg.use_open_clip

        if cfg.use_open_clip:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.open_clip_model,
                pretrained=cfg.open_clip_pretrained,
                device=device,
            )
            self.open_clip_model = model
            self._open_clip_preprocess = preprocess
            for p in self.open_clip_model.parameters():
                p.requires_grad = False
            self.open_clip_model.eval()
        else:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

            vm = CLIPVisionModelWithProjection.from_pretrained(cfg.model_id)
            self.hf_model = vm
            self.hf_model.to(device)
            for p in self.hf_model.parameters():
                p.requires_grad = False
            self.hf_model.eval()
            self.processor = CLIPImageProcessor.from_pretrained(cfg.model_id)

        self._eval_transform = transforms.Compose(
            [
                transforms.Resize(
                    cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(cfg.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _pixel_values_from_pil(self, images: List[Image.Image]) -> torch.Tensor:
        if self.use_open_clip:
            batch = torch.stack(
                [self._open_clip_preprocess(im.convert("RGB")) for im in images]
            )
            return batch.to(self.device, non_blocking=True)
        assert self.processor is not None
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device, non_blocking=True)

    @torch.inference_mode()
    def forward_image_tensor(
        self, pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if self.use_open_clip:
            assert self.open_clip_model is not None
            emb = self.open_clip_model.encode_image(pixel_values)
            emb = F.normalize(emb, dim=-1)
            return {"cls_final": emb}

        assert self.hf_model is not None
        outputs = self.hf_model(pixel_values=pixel_values, output_hidden_states=True)
        hs = outputs.hidden_states
        last = hs[-1]
        cls = outputs.image_embeds
        patch = last[:, 1:, :]
        cls = F.normalize(cls, dim=-1)

        out_layers: Dict[int, torch.Tensor] = {}
        out_patches: Dict[int, torch.Tensor] = {}
        if self.cfg.layers:
            for lb in self.cfg.layers:
                if lb < 0 or lb >= len(hs):
                    continue
                h = hs[lb]
                out_layers[int(lb)] = h[:, 0, :]
                if self.cfg.return_patch_layers:
                    out_patches[int(lb)] = h[:, 1:, :]

        result: Dict[str, torch.Tensor] = {
            "cls_final": cls,
            "patch_tokens_final": patch,
            "layer_cls": out_layers,
        }
        if self.cfg.return_patch_layers:
            result["layer_patch"] = out_patches
        return result

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
        for start in tqdm(range(0, len(image_paths), batch_size), desc="CLIP extract"):
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(Path(p)).convert("RGB") for p in batch_paths]
            pixel = self._pixel_values_from_pil(images)
            feats = self.forward_image_tensor(pixel)

            for i, p in enumerate(batch_paths):
                entry: Dict[str, torch.Tensor] = {}
                if token == "cls":
                    entry["cls_final"] = feats["cls_final"][i].detach().cpu()
                else:
                    if "patch_tokens_final" not in feats:
                        raise ValueError(
                            "patch_mean token requires HF CLIP (disable use_open_clip)."
                        )
                    patch = feats["patch_tokens_final"][i]
                    entry["patch_mean"] = patch.mean(dim=0).detach().cpu()

                if "layer_cls" in feats and feats["layer_cls"]:
                    entry["layer_cls"] = {
                        k: v[i].detach().cpu() for k, v in feats["layer_cls"].items()
                    }
                results[p] = entry
        return results

"""Frozen SigLIP 2 image/text encoders (transformers)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SiglipModel, SiglipProcessor


class SigLIPBackbone(nn.Module):
    """Wraps a pretrained SigLIP model.

    - :meth:`encode_image` returns the patch grid ``(B, visual_dim, side, side)``.
    - :meth:`encode_text` returns the pooled text embedding ``(B, text_dim)``.
    - :attr:`processor` is the matching ``SiglipProcessor`` for tokenization.
    """

    def __init__(self, backbone: str = "google/siglip2-base-patch16-256",
                 freeze: bool = True):
        super().__init__()
        self.encoder = SiglipModel.from_pretrained(backbone)
        self.processor = SiglipProcessor.from_pretrained(backbone)
        cfg = self.encoder.config
        self.image_size = cfg.vision_config.image_size
        self.patch_size = cfg.vision_config.patch_size
        self.visual_dim = cfg.vision_config.hidden_size
        self.text_dim = cfg.text_config.hidden_size
        self.text_length = cfg.text_config.max_position_embeddings
        if freeze:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # keep a frozen backbone in eval mode
        return self

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # SigLIP patch tokens carry no CLS token: L == side*side.
        out = self.encoder.vision_model(pixel_values=pixel_values).last_hidden_state
        side = self.image_size // self.patch_size
        return out.transpose(1, 2).reshape(out.shape[0], -1, side, side)

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask).pooler_output

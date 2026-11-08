"""SigLIP 2 image/text encoders with optional freezing."""
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    SiglipImageProcessor,
    SiglipModel,
    SiglipProcessor,
)


def load_siglip_processor(path):
    """Load the image processor and checkpoint-declared tokenizer."""
    return SiglipProcessor(
        image_processor=SiglipImageProcessor.from_pretrained(path),
        tokenizer=AutoTokenizer.from_pretrained(path),
    )


class SigLIPBackbone(nn.Module):
    """Pretrained SigLIP image/text encoders."""

    def __init__(
        self,
        backbone: str = "google/siglip2-base-patch16-256",
        freeze: bool = True,
    ):
        super().__init__()

        self.freeze = freeze
        self.encoder = SiglipModel.from_pretrained(backbone)
        self.processor = load_siglip_processor(backbone)

        cfg = self.encoder.config
        self.image_size = cfg.vision_config.image_size
        self.patch_size = cfg.vision_config.patch_size
        self.visual_dim = cfg.vision_config.hidden_size
        self.text_dim = cfg.text_config.hidden_size
        self.text_length = cfg.text_config.max_position_embeddings

        self.encoder.requires_grad_(not self.freeze)

        if self.freeze:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        # Frozen encoders stay in eval mode. Otherwise follow the caller.
        if self.freeze:
            self.encoder.eval()

        return self

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze else nullcontext()

        with context:
            out = self.encoder.vision_model(
                pixel_values=pixel_values
            ).last_hidden_state

            # SigLIP has spatial patch tokens without a CLS token.
            side = self.image_size // self.patch_size
            return out.transpose(1, 2).reshape(
                out.shape[0], -1, side, side
            )

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = torch.no_grad() if self.freeze else nullcontext()

        with context:
            return self.encoder.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).pooler_output

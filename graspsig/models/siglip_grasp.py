"""Full SiglipGrasp model: frozen SigLIP 2 backbone + FiLM-modulated head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SigLIPBackbone
from .grasp_head import GraspHead
from ..geometry import decode_maps


def activate(raw: torch.Tensor) -> torch.Tensor:
    """Map raw logits to activated maps: sigmoid(Q), tanh(sin/cos), softplus(W/H)."""
    return torch.cat([raw[:, :1].sigmoid(), raw[:, 1:3].tanh(),
                      F.softplus(raw[:, 3:5])], dim=1)


class SiglipGrasp(nn.Module):
    """Language-Conditioned Grasp Detection (SigLIP 2 + FiLM).

    ``forward`` returns raw logits ``(B, 5, S, S)``. :meth:`decode` turns
    activated maps into per-image ``(predictions, scores)``.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.image_size = m["image_size"]
        self.fusion = m["fusion"]
        self.backbone = SigLIPBackbone(m["backbone"], m["freeze"])
        self.head = GraspHead(m["visual_dim"], m["text_dim"], m["channels"], m["fusion"])
        # Self-describing spec so a checkpoint can reconstruct the model.
        self.spec = {k: m[k] for k in ("backbone", "fusion", "channels", "image_size",
                                        "visual_dim", "text_dim", "text_length")}

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.encoder.eval()
        return self

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        feature = self.backbone.encode_image(pixel_values)
        text = None
        if self.fusion != "none":
            if input_ids is None:
                raise ValueError("Language fusion requires tokenized instructions")
            text = self.backbone.encode_text(input_ids, attention_mask)
        raw = self.head(feature, text)
        return F.interpolate(raw, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)

    @torch.no_grad()
    def decode(self, maps: torch.Tensor, top_k: int = 5, valid_mask=None,
               radius: int | None = None):
        """Decode activated ``(B, 5, S, S)`` maps into per-sample (preds, scores)."""
        results = []
        for b in range(maps.shape[0]):
            preds, scores = decode_maps(
                maps[b].float().cpu().numpy(), top_k,
                None if valid_mask is None else valid_mask[b].cpu().numpy(), radius)
            results.append((preds, scores))
        return results

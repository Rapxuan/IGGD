"""SigLIP 2 backbone with a language-conditioned dense grasp head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SigLIPBackbone
from .grasp_head import GraspHead
from ..geometry import decode_maps


def activate(raw: torch.Tensor) -> torch.Tensor:
    """Activate Q, sin/cos and normalized grasp dimensions."""
    return torch.cat(
        [
            raw[:, :1].sigmoid(),
            raw[:, 1:3].tanh(),
            F.softplus(raw[:, 3:5]),
        ],
        dim=1,
    )


class SiglipGrasp(nn.Module):
    """Language-conditioned grasp detection.

    forward() returns raw logits with shape (B, 5, S, S).
    """

    def __init__(self, cfg: dict):
        super().__init__()

        # Copy to avoid changing the caller's config.
        m = dict(cfg["model"])

        # Historical checkpoints omitted this field.
        m.setdefault("freeze", True)

        self.image_size = m["image_size"]
        self.fusion = m["fusion"]

        self.backbone = SigLIPBackbone(
            backbone=m["backbone"],
            freeze=m["freeze"],
        )

        self.head = GraspHead(
            m["visual_dim"],
            m["text_dim"],
            m["channels"],
            m["fusion"],
        )

        # Include freeze so loading preserves the training configuration.
        self.spec = {
            k: m[k]
            for k in (
                "backbone",
                "freeze",
                "fusion",
                "channels",
                "image_size",
                "visual_dim",
                "text_dim",
                "text_length",
            )
        }

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature = self.backbone.encode_image(pixel_values)

        text = None
        if self.fusion != "none":
            if input_ids is None:
                raise ValueError(
                    "Language fusion requires tokenized instructions"
                )

            text = self.backbone.encode_text(
                input_ids,
                attention_mask,
            )

        raw = self.head(feature, text)

        return F.interpolate(
            raw,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    @torch.no_grad()
    def decode(
        self,
        maps: torch.Tensor,
        top_k: int = 5,
        valid_mask=None,
        radius: int | None = None,
    ):
        """Decode activated maps into per-image predictions and scores."""
        results = []

        for b in range(maps.shape[0]):
            preds, scores = decode_maps(
                maps[b].float().cpu().numpy(),
                top_k,
                (
                    None
                    if valid_mask is None
                    else valid_mask[b].cpu().numpy()
                ),
                radius,
            )
            results.append((preds, scores))

        return results

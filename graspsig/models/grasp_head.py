"""Language fusion (none / concat / film) + decoder + dense grasp head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    groups = math.gcd(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(groups, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(groups, cout),
        nn.GELU(),
    )


class GraspHead(nn.Module):
    """Projects the visual patch grid, fuses the text embedding, upsamples and
    predicts a dense 5-channel map ``(Q, sin2theta, cos2theta, W, H)``."""

    def __init__(self, visual_dim: int, text_dim: int, channels: int = 128,
                 fusion: str = "film"):
        super().__init__()
        if fusion not in {"none", "concat", "film"}:
            raise ValueError("fusion must be none, concat, or film")
        if channels < 16 or channels % 16:
            raise ValueError("channels must be a positive multiple of 16")
        self.fusion = fusion
        self.visual_projection = nn.Conv2d(visual_dim, channels, 1)
        self.text_projection = nn.Linear(text_dim, channels)
        if fusion == "film":
            self.condition = nn.Linear(channels, channels * 2)
            # zero-init => FiLM starts as an identity mapping.
            nn.init.zeros_(self.condition.weight)
            nn.init.zeros_(self.condition.bias)
        elif fusion == "concat":
            self.condition = nn.Conv2d(channels * 2, channels, 1)
        self.refine = conv_block(channels, channels)
        self.upsample = nn.ModuleList([
            conv_block(channels, channels // 2),
            conv_block(channels // 2, channels // 4),
        ])
        self.head = nn.Conv2d(channels // 4, 5, 1)

    def forward(self, feature: torch.Tensor, text: torch.Tensor | None) -> torch.Tensor:
        feature = self.visual_projection(feature)
        if self.fusion != "none":
            if text is None:
                raise ValueError("Language fusion requires a text embedding")
            text = self.text_projection(text)
            if self.fusion == "film":
                gamma, beta = self.condition(text).chunk(2, dim=-1)
                feature = feature * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
            else:  # concat
                broadcast = text[:, :, None, None].expand(-1, -1, feature.shape[2], feature.shape[3])
                feature = self.condition(torch.cat([feature, broadcast], dim=1))
        feature = self.refine(feature)
        for block in self.upsample:
            feature = F.interpolate(feature, scale_factor=2, mode="bilinear", align_corners=False)
            feature = block(feature)
        return self.head(feature)

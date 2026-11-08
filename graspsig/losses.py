"""Training losses: balanced BCE (quality) + smooth-L1 (geometry)."""

import torch
import torch.nn.functional as F

from .models.siglip_grasp import activate


def grasp_loss(raw: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Loss over raw logits ``(B, 5, S, S)``.

    ``target`` holds labels (quality in [0,1], geometry in the activation
    range); ``mask`` (B, S, S) is 1 on supervised (positive) pixels.
    """
    qloss = F.binary_cross_entropy_with_logits(raw[:, 0], target[:, 0], reduction="none")
    positive = mask
    negative = 1 - mask
    dims = (1, 2)
    qloss = ((qloss * positive).sum(dims) / positive.sum(dims).clamp_min(1)
             + (qloss * negative).sum(dims) / negative.sum(dims).clamp_min(1)).mean()
    pred = activate(raw)
    errors = F.smooth_l1_loss(pred[:, 1:], target[:, 1:], reduction="none")
    regression = ((errors * mask[:, None]).sum((1, 2, 3)) /
                  (4 * mask.sum(dims)).clamp_min(1)).mean()
    total = qloss + regression
    return total, {"loss": float(total.detach()), "quality": float(qloss.detach()),
                   "regression": float(regression.detach())}

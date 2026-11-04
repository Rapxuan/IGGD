"""Evaluation metrics: per-sample grasp success and Top-k accuracy."""

from __future__ import annotations

import math

import numpy as np

from .geometry import is_success


def topk_accuracy(all_preds, all_gts, top_k: int = 1, iou_thr: float = 0.25,
                  angle_tol_deg: float = 30.0) -> float:
    """Fraction of samples where any of the top-k predictions matches a GT grasp.

    ``all_preds`` / ``all_gts``: lists over samples of ``[x,y,w,h,theta]`` arrays.
    """
    if not all_preds:
        return 0.0
    angle_tol = math.radians(angle_tol_deg)
    correct = sum(
        1.0 if is_success(p, g, top_k, iou_thr, angle_tol) else 0.0
        for p, g in zip(all_preds, all_gts)
    )
    return correct / len(all_preds)


def success_at_k(all_preds, all_gts, k: int, iou_thr: float = 0.25,
                 angle_tol_deg: float = 30.0) -> float:
    return topk_accuracy(all_preds, all_gts, top_k=k, iou_thr=iou_thr,
                         angle_tol_deg=angle_tol_deg)

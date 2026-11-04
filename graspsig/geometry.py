"""Rotated-rectangle geometry, letterbox transform, dense targets and decoding.

Coordinates: x right, y down, ``theta`` clockwise radians modulo pi.
A grasp is ``[x, y, w, h, theta]`` with ``w`` along ``(cos theta, sin theta)``
and ``h`` perpendicular.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


def wrap_angle(theta: float) -> float:
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def angle_distance(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))


def corners(grasp) -> np.ndarray:
    x, y, w, h, a = map(float, grasp)
    u = np.array([math.cos(a), math.sin(a)]) * w / 2
    v = np.array([-math.sin(a), math.cos(a)]) * h / 2
    center = np.array([x, y])
    return np.stack([center - u - v, center + u - v,
                     center + u + v, center - u + v])


def _inside(grasp, xx, yy, compact: bool = False):
    x, y, w, h, a = grasp
    dx, dy = xx - x, yy - y
    u = dx * math.cos(a) + dy * math.sin(a)
    v = -dx * math.sin(a) + dy * math.cos(a)
    half_w = w / (6 if compact else 2)
    return (np.abs(u) <= half_w + 1e-7) & (np.abs(v) <= h / 2 + 1e-7)


def raster_iou(a, b) -> float:
    """Rotated-rectangle IoU on the integer pixel lattice (not AABB IoU)."""
    pts = np.concatenate([corners(a), corners(b)])
    lo = np.ceil(pts.min(axis=0) - 1e-7).astype(int)
    hi = np.floor(pts.max(axis=0) + 1e-7).astype(int)
    if np.any(hi < lo):
        return 0.0
    if np.prod(hi - lo + 1) > 16_000_000:
        raise ValueError("Rectangle union too large; check coordinate units.")
    yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
    ma, mb = _inside(a, xx, yy), _inside(b, xx, yy)
    union = np.count_nonzero(ma | mb)
    return float(np.count_nonzero(ma & mb) / union) if union else 0.0


def is_success(preds, gts, top_k: int = 1, iou_thr: float = 0.25,
               angle_tol: float = math.pi / 6) -> bool:
    for pred in preds[:top_k]:
        for gt in gts:
            if angle_distance(pred[4], gt[4]) <= angle_tol + 1e-8:
                if raster_iou(pred, gt) > iou_thr:
                    return True
    return False


@dataclass(frozen=True)
class Letterbox:
    """Aspect-preserving resize with zero padding to a square."""

    width: int
    height: int
    size: int

    @property
    def scale(self) -> float:
        return self.size / max(self.width, self.height)

    @property
    def offset(self) -> tuple[float, float]:
        s = self.scale
        return ((self.size - 1 - s * (self.width - 1)) / 2,
                (self.size - 1 - s * (self.height - 1)) / 2)

    def image(self, image: Image.Image) -> Image.Image:
        s = self.scale
        ox, oy = self.offset
        affine = (1 / s, 0, 0.5 - (ox + 0.5) / s,
                  0, 1 / s, 0.5 - (oy + 0.5) / s)
        return image.transform((self.size, self.size), Image.Transform.AFFINE,
                               affine, Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    def grasps(self, grasps, inverse: bool = False) -> np.ndarray:
        result = np.asarray(grasps, dtype=np.float32).copy().reshape(-1, 5)
        ox, oy = self.offset
        if inverse:
            result[:, :2] = (result[:, :2] - [ox, oy]) / self.scale
            result[:, 2:4] /= self.scale
        else:
            result[:, :2] = result[:, :2] * self.scale + [ox, oy]
            result[:, 2:4] *= self.scale
        return result

    def content_mask(self) -> np.ndarray:
        yy, xx = np.mgrid[:self.size, :self.size]
        ox, oy = self.offset
        s = self.scale
        return ((xx >= ox - s / 2) & (xx <= ox + s * (self.width - 0.5)) &
                (yy >= oy - s / 2) & (yy <= oy + s * (self.height - 0.5)))


def make_targets(grasps, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian quality map + per-pixel geometry labels.

    Returns ``(maps, mask)`` where ``maps`` is ``(5, S, S)`` =
    ``(Q, sin2theta, cos2theta, W/S, H/S)`` and ``mask`` marks supervised pixels.
    """
    maps = np.zeros((5, size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    best = np.full((size, size), np.inf)
    yy, xx = np.mgrid[:size, :size]
    for grasp in grasps:
        x, y, w, h, a = map(float, grasp)
        u = (xx - x) * math.cos(a) + (yy - y) * math.sin(a)
        v = -(xx - x) * math.sin(a) + (yy - y) * math.cos(a)
        support = _inside(grasp, xx, yy, compact=True)
        if not support.any():
            support[int(np.clip(round(y), 0, size - 1)),
                    int(np.clip(round(x), 0, size - 1))] = True
        dist = (u / max(w / 6, 1)) ** 2 + (v / max(h / 6, 1)) ** 2
        q = np.exp(-0.5 * dist) * support
        maps[0] = np.maximum(maps[0], q)
        own = support & (dist < best)
        maps[1, own] = math.sin(2 * a)
        maps[2, own] = math.cos(2 * a)
        maps[3, own] = w / size
        maps[4, own] = h / size
        mask[own] = 1
        best[own] = dist[own]
    return maps, mask


def decode_maps(maps, top_k: int = 5, valid_mask=None, radius=None):
    """Decode activated ``(5, S, S)`` maps into ``(predictions, scores)``."""
    maps = np.asarray(maps, dtype=np.float32)
    _, height, width = maps.shape
    if height != width or maps.shape[0] != 5 or not np.isfinite(maps).all():
        raise ValueError("Expected finite (5, S, S) prediction maps.")
    q = maps[0].copy()
    if valid_mask is not None:
        q[~valid_mask] = -np.inf
    radius = radius if radius is not None else max(2, width // 16)
    predictions, scores = [], []
    for _ in range(top_k):
        y, x = np.unravel_index(q.argmax(), q.shape)
        if not np.isfinite(q[y, x]):
            break
        angle = 0.5 * math.atan2(float(maps[1, y, x]), float(maps[2, y, x]))
        w, h = np.clip(maps[3:5, y, x] * width, 1, width * 2)
        predictions.append([float(x), float(y), float(w), float(h), angle])
        scores.append(float(q[y, x]))
        q[max(0, y - radius):y + radius + 1,
          max(0, x - radius):x + radius + 1] = -np.inf
    return predictions, scores

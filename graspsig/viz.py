"""Visualization helpers (PIL-based)."""

import numpy as np
from PIL import Image, ImageDraw

from .geometry import corners


def draw_grasps(image: Image.Image, grasps, text: str = "",
                ground_truth: bool = False) -> Image.Image:
    """Draw rotated-rectangle ``grasps`` (N, 5) onto a copy of ``image``."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    color = (0, 180, 0) if ground_truth else (255, 0, 0)
    for grasp in np.asarray(grasps, dtype=np.float32).reshape(-1, 5):
        pts = [(float(p[0]), float(p[1])) for p in corners(grasp)]
        draw.line(pts + [pts[0]], fill=color, width=3)
        draw.ellipse([grasp[0] - 2, grasp[1] - 2, grasp[0] + 2, grasp[1] + 2],
                     fill=(255, 255, 0))
    if text:
        draw.text((4, 4), text, fill=(255, 255, 255))
    return img

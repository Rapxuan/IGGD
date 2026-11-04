"""Configuration loading with a built-in default (works without PyYAML)."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CONFIG = {
    "model": {
        "backbone": "google/siglip2-base-patch16-256",
        "freeze": True,
        "image_size": 256,
        "patch_size": 16,
        "visual_dim": 768,
        "text_dim": 768,
        "text_length": 64,
        "channels": 128,
        "fusion": "film",
    },
    "dataset": {"max_boxes": 16, "num_workers": 4},
    "train": {
        "epochs": 30,
        "batch_size": 16,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "save_every": 5,
    },
    "eval": {"top_k": 5, "iou_threshold": 0.25, "angle_tolerance_deg": 30.0},
}


def load_config(path: str | Path | None) -> dict:
    """Load a YAML/JSON config, falling back to :data:`DEFAULT_CONFIG`."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text)
        except ImportError:
            return cfg
    else:
        data = json.loads(text)
    if data:
        _merge(cfg, data)
    return cfg


def _merge(base: dict, update: dict) -> None:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v

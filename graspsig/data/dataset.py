"""Grasp-Anything++ dataset loading (JSONL manifest) and batch collation."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..geometry import Letterbox, make_targets


def load_manifest(path: str | Path, root: str | Path) -> list[dict]:
    """Load a JSONL manifest and validate it.

    Each line: ``{"id", "scene_id", "image", "text", "grasps"}`` with
    ``grasps`` an Nx5 array of ``[x, y, w, h, theta]`` (theta clockwise radians).
    """
    root = Path(root).resolve()
    records, ids = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "scene_id", "image", "text", "grasps"}
            if not required.issubset(row):
                raise ValueError(f"{path}:{line_number}: missing {required - row.keys()}")
            for key in ("id", "scene_id", "image", "text"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f"{path}:{line_number}: {key} must be nonempty text")
            if row["id"] in ids:
                raise ValueError(f"Duplicate sample id: {row['id']}")
            ids.add(row["id"])
            image = (root / row["image"]).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError(f"Image missing or outside data root: {image}")
            grasps = np.asarray(row["grasps"], dtype=np.float32)
            if grasps.ndim != 2 or grasps.shape[1] != 5 or not len(grasps):
                raise ValueError(f"{row['id']}: expected nonempty Nx5 grasps")
            if not np.isfinite(grasps).all() or (grasps[:, 2:4] <= 0).any():
                raise ValueError(f"{row['id']}: invalid grasp values")
            if (np.abs(grasps[:, 4]) > np.pi / 2 + 1e-6).any():
                raise ValueError(f"{row['id']}: theta must be radians modulo pi")
            records.append(dict(row, _image=str(image)))
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def assert_disjoint(train: list[dict], val: list[dict]) -> None:
    scenes = {r["scene_id"] for r in train} & {r["scene_id"] for r in val}
    images = {r["_image"] for r in train} & {r["_image"] for r in val}
    if scenes or images:
        raise ValueError(f"Train/val leakage: {len(scenes)} shared scenes, "
                         f"{len(images)} shared images")


class GraspDataset(Dataset):
    def __init__(self, records: list[dict], image_size: int,
                 text_overrides: list[str] | None = None):
        self.records = records
        self.image_size = image_size
        self.text_overrides = text_overrides

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        with Image.open(row["_image"]) as source:
            image = source.convert("RGB")
        w, h = image.size
        raw = np.asarray(row["grasps"], dtype=np.float32)
        if ((raw[:, 0] < 0).any() or (raw[:, 0] > w - 1).any() or
                (raw[:, 1] < 0).any() or (raw[:, 1] > h - 1).any() or
                (raw[:, 2:4] > 2 * max(w, h)).any()):
            raise ValueError(f"{row['id']}: center/size outside image coordinate limits")
        transform = Letterbox(w, h, self.image_size)
        gt = transform.grasps(raw)
        targets, mask = make_targets(gt, self.image_size)
        text = self.text_overrides[index] if self.text_overrides is not None else row["text"]
        return {"id": row["id"], "image": transform.image(image), "text": text,
                "gt": gt, "targets": targets, "mask": mask,
                "valid": transform.content_mask()}


def process_inputs(processor, images: list, texts: list, max_length: int) -> dict:
    """Tokenize texts and normalize (already letterboxed) images."""
    pixels = processor.image_processor(images=images, do_resize=False, return_tensors="pt")
    tokens = processor.tokenizer(texts, padding="max_length", max_length=max_length,
                                 truncation=True, return_tensors="pt")
    return {**dict(pixels), **dict(tokens)}


def collate_batch(processor, max_length: int = 64):
    def _collate(items: list[dict]) -> dict:
        inputs = process_inputs(processor, [v["image"] for v in items],
                                [v["text"] for v in items], max_length)
        return {"inputs": dict(inputs),
                "ids": [v["id"] for v in items],
                "targets": torch.from_numpy(np.stack([v["targets"] for v in items])),
                "mask": torch.from_numpy(np.stack([v["mask"] for v in items])),
                "gt": [v["gt"] for v in items],
                "valid": [v["valid"] for v in items]}
    return _collate

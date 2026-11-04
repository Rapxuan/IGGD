"""Generate a tiny synthetic dataset (red/blue objects) for pipeline smoke tests."""

# from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from ..geometry import corners
from .grasp_anything_split import group_grasp_anything_records_by_split


def generate(out: str | Path, scenes: int = 12) -> Path:
    """Write ``scenes`` synthetic scenes + an ``all.jsonl`` manifest.

    This data only exercises the code path; it is not for paper results.
    """
    if scenes < 3:
        raise ValueError("Need at least 3 synthetic scenes")
    out = Path(out)
    images = out / "image"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(scenes):
        image = Image.new("RGB", (96, 64), "#333333")
        draw = ImageDraw.Draw(image)
        for obj, (color, cx) in enumerate((("red", 23), ("blue", 71))):
            cy = 26 + (index * 3 + obj * 7) % 15
            angle = math.radians((index % 3 - 1) * 12)
            body = [cx, cy, 24, 16, angle]
            draw.polygon([tuple(p) for p in corners(body)], fill=color)
            rows.append({"id": f"synthetic_{index}_{obj}", "scene_id": f"synthetic_{index}",
                         "image": f"image/{index:04d}.png", "text": f"Grasp the {color} object.",
                         "grasps": [[cx, cy, 18, 10, angle]]})
        image.save(images / f"{index:04d}.png")
    manifest = out / "all.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return manifest


def split_manifest(manifest: str | Path, data_root: str | Path, out: str | Path,
                   val_fraction: float = 0.1, test_fraction: float = 0.1,
                   seed: int = 42,
                   grasp_anything_split_dir: str | Path | None = None) -> dict:
    """Split by scene, optionally preserving Grasp-Anything seen/unseen membership.

    With Grasp-Anything IDs, only the training pool is split into train and val;
    test_fraction applies exclusively to the original custom split mode.
    """
    import random
    from .dataset import load_manifest

    if not 0 < val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    if grasp_anything_split_dir is None and not 0 < test_fraction < 1:
        raise ValueError("test-fraction must be between 0 and 1")
    rows = load_manifest(manifest, data_root)
    if grasp_anything_split_dir is None:
        subsets = {}
        training_pool = rows
        report = {"kind": "custom scene-disjoint split (not Grasp-Anything seen/unseen)"}
    else:
        subsets, report = group_grasp_anything_records_by_split(rows, grasp_anything_split_dir)
        training_pool = subsets["train"]
    scenes = sorted({r["scene_id"] for r in training_pool})
    random.Random(seed).shuffle(scenes)
    nval = max(1, round(len(scenes) * val_fraction))
    ntest = max(1, round(len(scenes) * test_fraction)) if grasp_anything_split_dir is None else 0
    if nval + ntest >= len(scenes):
        raise ValueError("Need more matched training scenes or smaller split fractions")
    groups = {"train": set(scenes[nval + ntest:]), "val": set(scenes[:nval])}
    if grasp_anything_split_dir is None:
        groups["test"] = set(scenes[nval:nval + ntest])
    for name, scenes_in_split in groups.items():
        subsets[name] = [r for r in training_pool if r["scene_id"] in scenes_in_split]
    out = Path(out)
    if Path(manifest).resolve() in {(out / f"{name}.jsonl").resolve() for name in subsets}:
        raise ValueError("Split output must not overwrite the input manifest")
    out.mkdir(parents=True, exist_ok=True)
    report.update({"seed": seed, "val_fraction": val_fraction, "validation_unit": "scene"})
    if grasp_anything_split_dir is None:
        report["test_fraction"] = test_fraction
    for name, subset in subsets.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for r in subset:
                stream.write(json.dumps({k: v for k, v in r.items()
                                         if not k.startswith("_")}) + "\n")
        report[name] = {"scenes": len({r["scene_id"] for r in subset}),
                        "samples": len(subset)}
    (out / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report

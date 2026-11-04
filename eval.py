"""Evaluate a checkpoint: Top-1/Top-5 grasp success, with optional text perturbation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import SiglipProcessor

from graspsig.config import load_config
from graspsig.data import load_manifest
from graspsig.engine import choose_device, evaluate, load_checkpoint, make_loader, seed_everything


def corrupt_text(records, mode: str, seed: int):
    overrides = [r["text"] for r in records]
    if mode == "empty":
        overrides = ["" for _ in records]
    elif mode == "shuffle":
        texts = [r["text"] for r in records]
        random.Random(seed).shuffle(texts)
        overrides = texts
    changed = (sum(a != b for a, b in zip(overrides, (r["text"] for r in records)))
               / len(records)) if records else 0.0
    return overrides, changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--text-mode", choices=["normal", "empty", "shuffle"], default="normal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    processor = SiglipProcessor.from_pretrained(args.checkpoint.parent / "processor")
    records = load_manifest(args.manifest, args.data_root)
    overrides, changed = corrupt_text(records, args.text_mode, args.seed)
    batch = args.batch_size or cfg["train"]["batch_size"]
    loader = make_loader(records, model, processor, batch, args.workers,
                         text_overrides=overrides)
    metrics = evaluate(model, loader, device, cfg["eval"]["iou_threshold"],
                       cfg["eval"]["angle_tolerance_deg"])
    metrics.update({"text_mode": args.text_mode, "changed_text_fraction": changed,
                    "checkpoint": str(args.checkpoint), "manifest": str(args.manifest)})
    content = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()

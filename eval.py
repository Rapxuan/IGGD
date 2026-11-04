"""Evaluate a checkpoint: Top-1/Top-5 grasp success, with optional text perturbation.

Two modes:
  * ``--manifest <file>``            evaluate one manifest (as before)
  * ``--both-splits <dir>``          evaluate ``<dir>/test_seen.jsonl`` and
                                     ``<dir>/test_unseen.jsonl`` and print two rows
                                     for the paper table (written by ``prepare_data.py split``)
"""

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


def run_one(args, cfg, model, processor, manifest: Path, device, text_mode: str) -> dict:
    """Evaluate one manifest and return its metrics."""
    records = load_manifest(manifest, args.data_root)
    overrides, changed = corrupt_text(records, text_mode, args.seed)
    batch = args.batch_size or cfg["train"]["batch_size"]
    loader = make_loader(records, model, processor, batch, args.workers,
                         text_overrides=overrides)
    metrics = evaluate(model, loader, device, cfg["eval"]["iou_threshold"],
                       cfg["eval"]["angle_tolerance_deg"])
    metrics.update({"text_mode": text_mode, "changed_text_fraction": changed,
                    "checkpoint": str(args.checkpoint), "manifest": str(manifest)})
    return metrics


def print_metric_rows(results: dict) -> None:
    """Print a compact two-row table copied straight into the paper."""
    header = f"{'split':<12}{'top1':>8}{'top5':>8}{'samples':>9}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(f"{name:<12}{metrics['top1_accuracy']:>8.4f}"
              f"{metrics['top5_accuracy']:>8.4f}{metrics['samples']:>9d}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", type=Path,
                   help="Single manifest to evaluate (omit if using --both-splits)")
    p.add_argument("--both-splits", type=Path, metavar="DIR",
                   help="Evaluate DIR/test_seen.jsonl and DIR/test_unseen.jsonl, printing two rows")
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--text-mode", choices=["normal", "empty", "shuffle"], default="normal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    if args.manifest is None and args.both_splits is None:
        p.error("either --manifest FILE or --both-splits DIR is required")
    if args.manifest is not None and args.both_splits is not None:
        p.error("use --manifest or --both-splits, not both")

    cfg = load_config(args.config)
    seed_everything(args.seed)
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    processor = SiglipProcessor.from_pretrained(args.checkpoint.parent / "processor")

    if args.both_splits is not None:
        results = {}
        for name in ("test_seen", "test_unseen"):
            manifest = args.both_splits / f"{name}.jsonl"
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"{manifest} not found; run `prepare_data.py split "
                    f"--grasp-anything-split ... --out {args.both_splits}` first")
            results[name] = run_one(args, cfg, model, processor, manifest, device,
                                    args.text_mode)
        print_metric_rows(results)
        payload = {"checkpoint": str(args.checkpoint), "text_mode": args.text_mode,
                   "splits": results}
        content = json.dumps(payload, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content + "\n", encoding="utf-8")
        print(content)
        return

    metrics = run_one(args, cfg, model, processor, args.manifest, device, args.text_mode)
    content = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()

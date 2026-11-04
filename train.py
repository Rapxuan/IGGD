"""Train SiglipGrasp (frozen SigLIP 2 + FiLM head).

Example::

    python train.py --config configs/default.yaml \\
        --train data/manifests/train.jsonl --val data/manifests/val.jsonl \\
        --data-root data/raw --out runs/film --fusion film --amp --amp-dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import SiglipProcessor

from graspsig.config import load_config
from graspsig.data import assert_disjoint, load_manifest
from graspsig.engine import (choose_device, evaluate, load_checkpoint, make_loader,
                             move_inputs, save_checkpoint, seed_everything)
from graspsig.losses import grasp_loss
from graspsig.models import SiglipGrasp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--val", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--backbone", default=None)
    p.add_argument("--fusion", choices=["none", "concat", "film"], default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="CUDA mixed precision")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--resume", type=Path)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.fusion:
        cfg["model"]["fusion"] = args.fusion
    if args.channels:
        cfg["model"]["channels"] = args.channels
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.workers is not None:
        cfg["dataset"]["num_workers"] = args.workers
    if args.amp_dtype != "float16" and not args.amp:
        p.error("--amp-dtype bfloat16 requires --amp")

    device = choose_device(args.device)
    seed_everything(args.seed)
    train_records = load_manifest(args.train, args.data_root)
    val_records = load_manifest(args.val, args.data_root)
    assert_disjoint(train_records, val_records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.resume:
        model, prev = load_checkpoint(args.resume, device)
        start, best = prev["epoch"] + 1, prev["best_top1"]
        processor = SiglipProcessor.from_pretrained(out / "processor")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        optimizer.load_state_dict(prev["optimizer"])
    else:
        if any((out / n).exists() for n in ("last.pt", "best.pt", "history.jsonl")):
            raise FileExistsError("Run already exists; choose a new --out or use --resume")
        model = SiglipGrasp(cfg).to(device)
        processor = model.backbone.processor
        processor.save_pretrained(out / "processor")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        start, best = 0, -1.0

    amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    if amp and amp_dtype == torch.bfloat16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This CUDA device does not support bfloat16")
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)

    (out / "config.json").write_text(json.dumps(
        {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "model_spec": model.spec}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "train_samples": len(train_records),
                      "val_samples": len(val_records), "fusion": model.fusion,
                      "amp": amp, "amp_dtype": args.amp_dtype,
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)},
                     flush=True))

    val_loader = make_loader(val_records, model, processor, cfg["train"]["batch_size"],
                             cfg["dataset"]["num_workers"])
    for epoch in range(start, cfg["train"]["epochs"]):
        model.train()
        loader = make_loader(train_records, model, processor, cfg["train"]["batch_size"],
                             cfg["dataset"]["num_workers"], shuffle=True,
                             seed=args.seed + epoch)
        totals, count = {}, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                raw = model(**move_inputs(batch["inputs"], device))
                loss, values = grasp_loss(raw.float(), batch["targets"].to(device),
                                          batch["mask"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            n = len(batch["ids"])
            count += n
            for k, v in values.items():
                totals[k] = totals.get(k, 0.0) + v * n
        metrics = evaluate(model, val_loader, device, cfg["eval"]["iou_threshold"],
                           cfg["eval"]["angle_tolerance_deg"])
        improved = metrics["top1_accuracy"] > best
        best = max(best, metrics["top1_accuracy"])
        report = {"epoch": epoch + 1, "train": {k: v / count for k, v in totals.items()},
                  "validation": metrics, "best_top1": best}
        with (out / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        state = {"epoch": epoch, "model_spec": model.spec, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "best_top1": best,
                 "scaler": scaler.state_dict(), "amp": amp, "amp_dtype": args.amp_dtype}
        save_checkpoint(out / "last.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

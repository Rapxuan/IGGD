"""Draw language-conditioned grasps on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import SiglipProcessor

from graspsig.data.dataset import process_inputs
from graspsig.engine import choose_device, load_checkpoint, move_inputs
from graspsig.geometry import Letterbox, decode_maps
from graspsig.models import activate
from graspsig.viz import draw_grasps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--text", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if not 1 <= args.top_k <= 5:
        p.error("top-k must be between 1 and 5")
    if args.output.resolve() == args.image.resolve():
        p.error("Use an output path different from the source image")

    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    processor = SiglipProcessor.from_pretrained(args.checkpoint.parent / "processor")

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    transform = Letterbox(*image.size, model.image_size)
    inputs = process_inputs(processor, [transform.image(image)], [args.text],
                            model.backbone.text_length)
    with torch.inference_mode():
        maps = activate(model(**move_inputs(inputs, device)))[0].float().cpu().numpy()
    grasps, scores = decode_maps(maps, args.top_k, transform.content_mask())
    original = transform.grasps(grasps, inverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw_grasps(image, original, args.text).save(args.output)
    result = {"image": str(args.image), "text": args.text, "grasps": original.tolist(),
              "scores": scores, "coordinates": "original image pixels; theta clockwise radians",
              "score_note": "predicted quality, not a calibrated physical grasp probability"}
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

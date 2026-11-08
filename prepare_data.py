"""Prepare data: inspect/convert Grasp-Anything++, split, preview, synthetic.

Convert expects the raw layout::

    <images>/<scene>.jpg
    <instructions>/<scene>_<object>_<part>.pkl
    <grasps>/<scene>_<object>_<part>.pt          # Nx6 [aux, x, y, w, h, theta_degrees]
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from graspsig.data import generate, load_manifest, split_manifest
from graspsig.geometry import wrap_angle
from graspsig.viz import draw_grasps


def select_instructions(value, key: str | None = None):
    """Refuse ambiguous/nested data rather than pairing the wrong text."""
    if key:
        for component in key.split("."):
            if isinstance(value, dict):
                value = value[component] if component in value else value[int(component)]
            elif isinstance(value, (list, tuple)):
                value = value[int(component)]
            else:
                raise ValueError(f"Cannot traverse instruction key {key}")
    elif isinstance(value, dict):
        candidates = [k for k in ("instruction", "instructions", "text", "prompt",
                                  "grasp_instructions") if k in value]
        if len(candidates) != 1:
            raise ValueError("Ambiguous instruction dict; inspect it and set --instruction-key")
        value = value[candidates[0]]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Expected one instruction or a nonempty list of strings; inspect the pickle")
    if not all(isinstance(s, str) and s.strip() for s in value):
        raise ValueError("Nested/mixed instructions: select the exact field with --instruction-key")
    return list(dict.fromkeys(s.strip() for s in value))


def load_raw_grasps(path: Path):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError(f"{path}: expected tensor Nx6 [aux,x,y,w,h,theta_degrees]")
    data = raw.detach().cpu().numpy().astype(np.float64)
    if not len(data) or not np.isfinite(data).all() or (data[:, 3:5] <= 0).any():
        raise ValueError(f"{path}: empty or invalid positive grasp labels")
    result = data[:, 1:].copy()
    # Official LGD negates degrees for its CCW angle; our public representation
    # is clockwise in image coordinates, so NO sign flip.
    result[:, 4] = wrap_angle(np.deg2rad(result[:, 4]))
    return result.tolist()


def _write_manifest(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_inspect(args):
    if args.file.suffix == ".pt":
        value = torch.load(args.file, map_location="cpu", weights_only=True)
    elif args.file.suffix == ".pkl":
        if not args.trust_pickle:
            raise ValueError("Inspecting pickle requires --trust-pickle (trusted source only)")
        with args.file.open("rb") as stream:
            value = pickle.load(stream)
    else:
        raise ValueError("Expected .pt or .pkl")

    def describe(v, depth=0):
        if depth >= 3:
            return {"type": type(v).__name__, "preview": repr(v)[:180]}
        if isinstance(v, dict):
            return {"type": "dict", "length": len(v),
                    "items": {str(k): describe(x, depth + 1) for k, x in list(v.items())[:8]}}
        if isinstance(v, (list, tuple)):
            return {"type": type(v).__name__, "length": len(v),
                    "items": [describe(x, depth + 1) for x in v[:5]]}
        if isinstance(v, (torch.Tensor, np.ndarray)):
            return {"type": type(v).__name__, "shape": list(v.shape), "preview": str(v)[:300]}
        return {"type": type(v).__name__, "preview": str(v)[:300]}

    print(json.dumps(describe(value), ensure_ascii=False, indent=2))


def cmd_convert(args):
    if not args.trust_pickle:
        raise ValueError("Only load trusted dataset pickles; add --trust-pickle after checking the source")
    root = args.root.resolve()
    files = sorted(args.grasps.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt labels in {args.grasps}")
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError("max-scenes must be positive")
    scene_ids = {f.stem.rsplit("_", 2)[0] for f in files}
    selected = set(sorted(scene_ids)[: args.max_scenes]) if args.max_scenes else scene_ids

    def records():
        for file in files:
            parts = file.stem.rsplit("_", 2)
            if len(parts) != 3 or not all(x.isdigit() for x in parts[1:]):
                raise ValueError(f"Expected scene_object_part.pt, got {file.name}")
            scene = parts[0]
            if scene not in selected:
                continue
            matches = [args.images / (scene + s) for s in (".jpg", ".png", ".jpeg")]
            matches = [p.resolve() for p in matches if p.is_file()]
            if len(matches) != 1:
                raise ValueError(f"{scene}: expected exactly one matching image")
            image = matches[0]
            if not image.is_relative_to(root):
                raise ValueError("Images must reside under --root")
            instruction_file = args.instructions / (file.stem + ".pkl")
            with instruction_file.open("rb") as stream:
                texts = select_instructions(pickle.load(stream), args.instruction_key)
            grasps = load_raw_grasps(file)
            for index, text in enumerate(texts):
                yield {"id": f"{file.stem}:{index}", "scene_id": scene,
                       "image": image.relative_to(root).as_posix(), "text": text,
                       "grasps": grasps}

    _write_manifest(args.output, records())
    converted = load_manifest(args.output, root)
    print(json.dumps({"samples": len(converted), "scenes": len(selected),
                      "manifest": str(args.output)}, indent=2))


def cmd_split(args):
    if args.grasp_anything_split_dir is not None and not args.trust_pickle:
        raise ValueError("Grasp-Anything .obj lists are pickles; add --trust-pickle for a trusted source")
    report = split_manifest(args.manifest, args.data_root, args.out,
                            args.val_fraction, args.test_fraction, args.seed,
                            grasp_anything_split_dir=args.grasp_anything_split_dir)
    print(json.dumps(report, indent=2))


def cmd_preview(args):
    rows = load_manifest(args.manifest, args.data_root)
    random.Random(args.seed).shuffle(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows[: args.count]):
        with Image.open(row["_image"]) as image:
            rendered = draw_grasps(image, row["grasps"], row["text"], ground_truth=True)
        rendered.save(Path(args.out) / f"{i:03d}.png")
    print(f"Wrote {min(args.count, len(rows))} annotation previews to {args.out}")


def cmd_synthetic(args):
    manifest = generate(args.out, args.scenes)
    print(f"Synthetic plumbing-test data: {manifest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("--file", required=True, type=Path)
    p.add_argument("--trust-pickle", action="store_true")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("convert")
    for name in ("images", "instructions", "grasps", "root", "output"):
        p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--instruction-key", help="Dotted path within each instruction pickle")
    p.add_argument("--trust-pickle", action="store_true")
    p.add_argument("--max-scenes", type=int)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("split")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1,
                   help="Custom split only; ignored with --grasp-anything-split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grasp-anything-split", dest="grasp_anything_split_dir", type=Path,
                   help="Grasp-Anything split/grasp-anything or LGD split/grasp-anything++ directory")
    p.add_argument("--trust-pickle", action="store_true",
                   help="Allow loading trusted Grasp-Anything .obj ID lists")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("preview")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("synthetic")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--scenes", type=int, default=12)
    p.set_defaults(func=cmd_synthetic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

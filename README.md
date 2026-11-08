# IGGD — Instruction-Guided Grasp Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)

Language-driven grasp detection: given an RGB image and a natural-language
instruction (*"grasp the blue bottle"*), predict the rotated-rectangle grasp
pose `(x, y, w, h, θ)` of the object the instruction refers to.

IGGD keeps a pretrained **SigLIP 2** dual encoder and trains a lightweight
**FiLM-conditioned** decoder that regresses a dense grasp map. Only the referred
object yields confident proposals, so "which object" and "where to grasp" are
solved by a single network instead of a detector-plus-grasp-generator pipeline.

## Method

1. **Frozen SigLIP 2** encoder (`google/siglip2-base-patch16-256`) encodes the
   image into a `16×16` patch grid (no CLS token — SigLIP patches are all
   spatial) and the instruction into one pooled text embedding.
2. **FiLM conditioning.** The text embedding is mapped to per-channel scale and
   shift and applied residually to the visual features:
   `z ← z ⊙ (1 + γ) + β`. The layer is zero-initialised, so it starts as an
   identity and learns to amplify the channels of the referred object.
   Two baselines are available: `none` (no language) and `concat`.
3. **Dense grasp head.** The decoder upsamples to input resolution and predicts,
   at every pixel, a quality logit `q̂` and
   `(sin 2θ̂, cos 2θ̂, Ŵ, Ĥ)`. The double-angle parameterisation makes the head
   invariant to the 180° symmetry of a grasp rectangle; `W, H` use softplus.
4. **Loss.** Foreground/background-balanced BCE on quality plus a smooth-L1 on
   geometry over positive pixels. Only the head trains by default; the backbone
   can optionally be fine-tuned at a 10× smaller learning rate.

## Results

Grasp-Anything++ official seen/unseen splits. A prediction is correct when its
rotated-rectangle IoU with a ground-truth grasp exceeds `0.25` and the angle
differs by at most `30°` (mod 180°).

| Method | Seen Top-1 | Seen Top-5 | Unseen Top-1 | Unseen Top-5 |
|---|---|---|---|---|
| **IGGD (frozen backbone)** | 0.58 | 0.84 | 0.38 | 0.68 |
| IGGD (full fine-tune) | 0.60 | 0.80 | 0.42 | 0.62 |

**Language dependency** (frozen backbone, seen split): shuffling the
instructions drops Top-1 from `0.58` to `0.31` (Top-5 `0.84 → 0.54`) and
emptying them drops it to `0.30` — the decoder genuinely relies on the text
signal rather than the image alone.

## Repository layout

```
IGGD/
├── configs/
│   ├── default.yaml        # full fine-tuning configuration
│   └── head_only.yaml      # frozen-backbone FiLM configuration
├── graspsig/
│   ├── config.py           # config loading + defaults
│   ├── geometry.py         # letterbox, dense targets, decode, rotated IoU
│   ├── metrics.py          # Top-k grasp success
│   ├── losses.py           # balanced BCE + smooth-L1
│   ├── viz.py              # draw grasp rectangles
│   ├── engine.py           # device/seed/loaders/checkpoints/eval
│   ├── data/
│   │   ├── dataset.py                  # JSONL manifest loader + collation
│   │   ├── grasp_anything_split.py     # seen/unseen object-ID grouping
│   │   └── synthetic.py                # synthetic data + scene split
│   └── models/
│       ├── backbone.py     # SigLIP 2 encoders (+ optional freezing)
│       ├── grasp_head.py   # fusion + decoder + head
│       └── siglip_grasp.py # full model + activation
├── prepare_data.py         # inspect / convert / split / preview / synthetic
├── train.py
├── eval.py
├── predict.py
├── split/                  # official Grasp-Anything++ ID lists (seen/unseen)
```

## Install

```bash
conda create -n iggd python=3.11 -y
conda activate iggd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8 build
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The SigLIP 2 weights (`google/siglip2-base-patch16-256`) are downloaded from the
Hugging Face Hub on first use. Set `HF_HUB_OFFLINE=1` if you keep a local mirror
and point `model.backbone` at that directory instead.

## Quick smoke test (synthetic data, no downloads)

```bash
python prepare_data.py synthetic --out data/synthetic --scenes 12
python prepare_data.py split --manifest data/synthetic/all.jsonl \
  --data-root data/synthetic --out data/synthetic/manifests

python train.py --config configs/head_only.yaml \
  --train data/synthetic/manifests/train.jsonl \
  --val data/synthetic/manifests/val.jsonl --data-root data/synthetic \
  --out runs/smoke --epochs 1 --batch-size 8
```

## Prepare Grasp-Anything++

Only images, instructions and positive grasps are used; negative labels and
part masks are not read. Arrange the downloads as:

```text
data/raw/
  grasp-anything/
    image/<scene>.jpg
  grasp-anything-pp/
    grasp_instructions/<scene>_<object_id>_<part_id>.pkl
    grasp_label_positive/<scene>_<object_id>_<part_id>.pt
```

Convert them into a single JSONL manifest:

```bash
python prepare_data.py convert \
  --images data/raw/grasp-anything/image \
  --instructions data/raw/grasp-anything-pp/grasp_instructions \
  --grasps data/raw/grasp-anything-pp/grasp_label_positive \
  --root data/raw --output data/manifests/all.jsonl --trust-pickle
```

Use `--instruction-key` if the instruction pickles carry a nested schema, and
`prepare_data.py inspect --file <pkl> --trust-pickle` to see what a record
contains. A converted record looks like:

```json
{"id": "scene0_0_0:0", "scene_id": "scene0",
 "image": "grasp-anything/image/scene0.jpg",
 "text": "Grasp the blue bottle.",
 "grasps": [[120, 150, 60, 25, 0.2]]}
```

`image` is relative to `--data-root`; `grasps` are `[x, y, w, h, θ]` in original
pixels with `θ` clockwise in radians.

### Seen/unseen splits

The official Grasp-Anything++ object-ID lists are already in `split/`:

```text
split/grasp-anything++/
  train/seen.obj
  test/seen.obj
  test/unseen.obj
```

Build the manifests from them:

```bash
python prepare_data.py split \
  --manifest data/manifests/all.jsonl --data-root data/raw \
  --grasp-anything-split split/grasp-anything++ \
  --out data/manifests/grasp-anything++ --val-fraction 0.1 --seed 42 --trust-pickle
```

| ID list | Output |
|---|---|
| `train/seen.obj` | `train.jsonl` + `val.jsonl` |
| `test/seen.obj` | `test_seen.jsonl` |
| `test/unseen.obj` | `test_unseen.jsonl` |

Matching follows LGD's object key: `scene_0_12:3` is annotation `scene_0_12`,
which belongs to Grasp-Anything object `scene_0`. The validation holdout is
drawn scene-disjoint from the *training* pool only; the test splits are never
used for model selection. `split_report.json` records source lists, seed,
sample/scene counts, `unmatched_samples` and `missing_objects`, so a partial
download is easy to spot. An object ID appearing in two source lists is
rejected rather than silently assigned.

Without `--grasp-anything-split`, a plain scene-disjoint 80/10/10 split is
produced instead (`--test-fraction` applies to that mode).

To eyeball annotations before training:

```bash
python prepare_data.py preview --manifest data/manifests/grasp-anything++/train.jsonl \
  --data-root data/raw --out runs/annotation_check --count 30
```

## Training

Frozen backbone (the main experiment):

```bash
python train.py --config configs/head_only.yaml \
  --train data/manifests/grasp-anything++/train.jsonl \
  --val   data/manifests/grasp-anything++/val.jsonl \
  --data-root data/raw --out runs/film --fusion film \
  --epochs 500 --batch-size 8 --amp --amp-dtype bfloat16
```

Full fine-tuning (`configs/default.yaml` sets `freeze: false`; use a smaller
backbone learning rate as in the paper):

```bash
python train.py --config configs/default.yaml \
  --train data/manifests/grasp-anything++/train.jsonl \
  --val   data/manifests/grasp-anything++/val.jsonl \
  --data-root data/raw --out runs/film_ft --fusion film \
  --epochs 80 --batch-size 16 --amp --amp-dtype bfloat16
```

Ablations — swap `--fusion` for the language-conditioning baseline:

```bash
python train.py ... --out runs/none   --fusion none    # no language
python train.py ... --out runs/concat --fusion concat  # concatenation instead of FiLM
```

Checkpoints are written to `--checkpoint-dir` (default `runs/checkpoints`)
as `epoch_XXXX.pt`, only after `--save-after` epochs and every `--save-every`
epochs thereafter. Each run also writes `config.json` and `history.jsonl` to
`--out`. Resume with `--resume <checkpoint>`; AMP settings must be unchanged.

## Evaluation

```bash
# both official splits in one run (prints the table above)
python eval.py --checkpoint runs/film/epoch_0500.pt \
  --both-splits data/manifests/grasp-anything++ --data-root data/raw \
  --output runs/film/test_metrics.json

# a single manifest
python eval.py --checkpoint runs/film/epoch_0500.pt \
  --manifest data/manifests/grasp-anything++/test_seen.jsonl \
  --data-root data/raw --output runs/film/test_seen_metrics.json

# language-dependency check: shuffled / empty instructions
python eval.py --checkpoint runs/film/epoch_0500.pt \
  --manifest data/manifests/grasp-anything++/test_seen.jsonl \
  --data-root data/raw --text-mode shuffle --output runs/film/shuffled.json
```

`--both-splits DIR` reads `DIR/test_seen.jsonl` and `DIR/test_unseen.jsonl`,
prints a compact `split / top1 / top5 / samples` table and writes a combined
JSON with a `"splits"` object.

## Prediction

```bash
python predict.py --checkpoint runs/film/epoch_0500.pt \
  --image data/raw/grasp-anything/image/SCENE.jpg \
  --text "Grasp the blue bottle." \
  --output runs/film/blue_bottle.png --top-k 1
```

## Notes

- Reported numbers are offline box-matching accuracy, **not** calibrated
  physical grasp success.
- SigLIP 2 and FiLM are existing methods — the contribution here is the task
  adaptation, the dense rotated-rectangle head and the fusion ablation.
- `data/`, `runs/` and weight files are git-ignored; this repository ships code,
  configs and the official split lists.

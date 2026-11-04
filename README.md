# grasp-sig-split — SigLIP 2 + FiLM with Grasp-Anything Dataset Splits

A single-network, **language-driven grasp detection** method for the
*"Language-Driven Grasp Detection"* programming test.

Given an RGB image and a natural-language instruction (e.g. *"grasp the blue
bottle"*), the model predicts the **rotated-rectangle grasp pose** of the
referred object — `(x, y, w, h, theta)` with `theta` clockwise in radians.

## Method (the "proposed method")

A **frozen SigLIP 2** encoder (`google/siglip2-base-patch16-256`) provides both
modalities:

1. The **vision tower** encodes the image into a `16x16` patch grid (no CLS
   token is removed — SigLIP patches are all spatial).
2. The **text tower** turns the instruction into one pooled embedding.
3. A small **head** projects the patch grid, fuses the text embedding through
   one of three strategies — `none` / `concat` / **`film`** (feature-wise linear
   modulation, `F' = F*(1+gamma) + beta`, zero-initialised as identity) — then
   upsamples and predicts a **dense** map at every pixel:
   `(Q, sin(2θ), cos(2θ), W, H)`.
4. The double-angle `sin(2θ)/cos(2θ)` makes the head invariant to the 180°
   symmetry of a grasp rectangle; `W,H` are activated with softplus.

Only the head trains; the backbone stays frozen. This is what makes the detector
"language-driven" while staying cheap enough for a free Colab GPU.

## Layout

```
grasp-sig-split/
├── configs/default.yaml      # model / dataset / train / eval settings
├── graspsig/
│   ├── config.py             # config loading + defaults
│   ├── geometry.py           # letterbox, dense targets, decode, rotated IoU
│   ├── metrics.py            # Top-k grasp success
│   ├── losses.py             # balanced BCE + smooth-L1
│   ├── viz.py                # draw rectangles
│   ├── engine.py             # device/seed/loaders/checkpoints/eval
│   ├── data/
│   │   ├── dataset.py        # JSONL manifest loader + collation
│   │   ├── grasp_anything_split.py  # seen/unseen object-ID grouping
│   │   └── synthetic.py      # synthetic data + scene split
│   └── models/
│       ├── backbone.py       # frozen SigLIP 2 encoders
│       ├── grasp_head.py     # fusion + decoder + head
│       └── siglip_grasp.py   # full model + activation
├── prepare_data.py           # synthetic / split / preview
├── train.py
├── eval.py
├── predict.py
└── paper/                    # 2-page CVPR paper (LaTeX)
```

## Install

```bash
conda create -n wzx-grasp python=3.11 -y
conda activate wzx-grasp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 5090 / Blackwell
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Quick smoke test (synthetic, no real data)

```bash
python prepare_data.py synthetic --out data/synthetic --scenes 12
python prepare_data.py split --manifest data/synthetic/all.jsonl \
  --data-root data/synthetic --out data/synthetic/manifests

python train.py --train data/synthetic/manifests/train.jsonl \
  --val data/synthetic/manifests/val.jsonl --data-root data/synthetic \
  --out runs/smoke --fusion film --epochs 1 --batch-size 8
```

## Prepare real Grasp-Anything++ data

Keep the two downloaded datasets separate; only the following directories are
needed by this model. Neither negative grasp labels nor part masks are read.

```text
data/raw/
  grasp-anything/
    image/<scene>.jpg
  grasp-anything-pp/
    grasp_instructions/<scene>_<object_id>_<part_id>.pkl
    grasp_label_positive/<scene>_<object_id>_<part_id>.pt
```

Convert the paired images, instructions and positive grasps into one manifest:

```bash
python prepare_data.py convert \
  --images data/raw/grasp-anything/image \
  --instructions data/raw/grasp-anything-pp/grasp_instructions \
  --grasps data/raw/grasp-anything-pp/grasp_label_positive \
  --root data/raw --output data/manifests/all.jsonl --trust-pickle
```

If the instruction pickle has a nested schema, select its exact field with
`--instruction-key`. A converted record has this format:

```json
{"id":"scene0_0_0:0","scene_id":"scene0","image":"grasp-anything/image/scene0.jpg",
 "text":"Grasp the blue bottle.","grasps":[[120,150,60,25,0.2]]}
```

`image` is relative to `--data-root`; `grasps` are `[x,y,w,h,theta]` in original
pixels, theta clockwise radians. All commands use the same `data/raw` root.

### Reuse Grasp-Anything / LGD seen/unseen IDs

Obtain the lists from the [LGD repository](https://github.com/Fsoft-AIC/LGD/tree/501904dc5c30c8fad58d90f7e4f49b4095f11eb0/split/grasp-anything%2B%2B),
preserving this directory structure:

```text
LGD/split/grasp-anything++/
  train/seen.obj
  test/seen.obj
  test/unseen.obj
```

Use the existing `split` command with one additional path:

```bash
python prepare_data.py split \
  --manifest data/manifests/all.jsonl --data-root data/raw \
  --grasp-anything-split LGD/split/grasp-anything++ \
  --out data/manifests/grasp-anything++ --val-fraction 0.1 --seed 42 --trust-pickle
```

The `.obj` files are trusted pickle ID lists. Matching follows LGD's object
key: `scene_0_12:3` identifies annotation `scene_0_12`, which matches the Grasp-Anything
object ID `scene_0`. Part IDs may have multiple digits. Instructions and parts
of the same object inherit its Grasp-Anything split membership. For handwritten manifests,
use the same sample-ID format or supply `"annotation_id": "scene_0_12"`.

ID matching lives in `graspsig/data/grasp_anything_split.py`:

| Function | Purpose |
|---|---|
| `load_grasp_object_ids` | Read object IDs from a `.obj` list |
| `extract_grasp_object_id` | Extract `scene_object` from a manifest record |
| `group_grasp_anything_records_by_split` | Group records by the source train/seen/unseen lists |

| Grasp-Anything / LGD ID list | Output |
|---|---|
| `train/seen.obj` | `train.jsonl` and `val.jsonl` |
| `test/seen.obj` | `test_seen.jsonl` |
| `test/unseen.obj` | `test_unseen.jsonl` |

Grasp-Anything / LGD train/test and seen/unseen membership is preserved. Only the source
training pool gets a custom, scene-disjoint validation holdout (10% of its
scenes by default); this holdout is not LGD's annotation-index train/val split.
Grasp-Anything / LGD test sets are never used to select the model. Their scene/image
overlap with other sets is not changed: object-ID membership does not imply
scene-disjoint Grasp-Anything / LGD test sets.

`split_report.json` records the source lists, seed, sample/scene counts and
object-ID coverage under `grasp_anything_split_dir` and `grasp_anything_id_lists`.
Samples absent from all supplied lists are excluded and
counted as `unmatched_samples`; listed IDs absent from the local manifest are
counted as `missing_objects`. Incomplete local downloads can therefore produce
partial or empty test manifests. An object ID present in multiple Grasp-Anything / LGD
lists is rejected instead of being placed in conflicting sets.

For the original Grasp-Anything layout, point `--grasp-anything-split` at its
`split/grasp-anything` directory containing only `seen.obj` and `unseen.obj`.
The seen pool becomes train/val and unseen becomes `test_unseen.jsonl`; no
independent `test_seen.jsonl` is invented. Choose the Grasp-Anything / LGD lists appropriate
to the annotation version you downloaded. The model/training code still reads
JSONL files and does not scan `.obj` lists automatically.

Without `--grasp-anything-split`, the existing custom scene-disjoint 80/10/10 split
remains available. `--test-fraction` applies only to that custom mode.

To render annotation previews:

```bash
python prepare_data.py preview --manifest data/manifests/grasp-anything++/train.jsonl \
  --data-root data/raw --out runs/annotation_check --count 30
```

## Train (main experiment + ablation)

```bash
python train.py --train data/manifests/grasp-anything++/train.jsonl --val data/manifests/grasp-anything++/val.jsonl \
  --data-root data/raw --out runs/film --fusion film \
  --epochs 30 --batch-size 16 --amp --amp-dtype bfloat16

python train.py --train data/manifests/grasp-anything++/train.jsonl --val data/manifests/grasp-anything++/val.jsonl \
  --data-root data/raw --out runs/none --fusion none --epochs 30 --batch-size 16 --amp --amp-dtype bfloat16

python train.py --train data/manifests/grasp-anything++/train.jsonl --val data/manifests/grasp-anything++/val.jsonl \
  --data-root data/raw --out runs/concat --fusion concat --epochs 30 --batch-size 16 --amp --amp-dtype bfloat16
```

## Evaluate & predict

```bash
python eval.py --checkpoint runs/film/best.pt --manifest data/manifests/grasp-anything++/test_seen.jsonl \
  --data-root data/raw --output runs/film/test_seen_metrics.json

python eval.py --checkpoint runs/film/best.pt --manifest data/manifests/grasp-anything++/test_unseen.jsonl \
  --data-root data/raw --output runs/film/test_unseen_metrics.json

# language-dependency check: shuffled/empty instructions should hurt
python eval.py --checkpoint runs/film/best.pt --manifest data/manifests/grasp-anything++/test_seen.jsonl \
  --data-root data/raw --text-mode shuffle --output runs/film/shuffled_metrics.json

python predict.py --checkpoint runs/film/best.pt --image data/raw/grasp-anything/image/SCENE.jpg \
  --text "Grasp the blue bottle." --output runs/film/blue_bottle.png --top-k 1
```

Evaluation reports **Top-1/Top-5 grasp success**: a prediction is correct when
its rotated-rectangle IoU with a ground-truth grasp exceeds `0.25` and the angle
difference is within `30°` (mod 180°).

## Notes

- Metric is offline box-matching accuracy, **not** calibrated physical grasp
  success.
- `split --grasp-anything-split ...` reuses Grasp-Anything / LGD ID membership and holds validation
  out of the training pool by scene. Plain `split` uses custom 80/10/10 scenes.
- FiLM and SigLIP are existing methods — cite them; your contribution is the
  task adaptation and the fusion ablation.

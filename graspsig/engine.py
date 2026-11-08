"""Training/eval utilities: device, seeding, loaders, checkpoints, evaluation."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GraspDataset, collate_batch
from .geometry import decode_maps
from .metrics import topk_accuracy
from .models.siglip_grasp import SiglipGrasp, activate


def choose_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_inputs(inputs: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def make_loader(records, model, processor, batch_size, workers,
                shuffle: bool = False, seed: int = 0,
                text_overrides: list[str] | None = None) -> DataLoader:
    dataset = GraspDataset(records, model.image_size, text_overrides)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, generator=generator,
                      collate_fn=collate_batch(processor, model.backbone.text_length))


def evaluate(model, loader, device, iou_thr: float = 0.25,
             angle_tol_deg: float = 30.0) -> dict:
    """Top-1/Top-5 grasp success on a loader (each pred matches the GT only)."""
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch in loader:
            raw = model(**move_inputs(batch["inputs"], device))
            maps = activate(raw)
            for b in range(raw.shape[0]):
                preds, _ = decode_maps(maps[b].float().cpu().numpy(), top_k=5,
                                       valid_mask=batch["valid"][b])
                all_preds.append(preds)
                all_gts.append(batch["gt"][b])
    return {
        "top1_accuracy": topk_accuracy(all_preds, all_gts, 1, iou_thr, angle_tol_deg),
        "top5_accuracy": topk_accuracy(all_preds, all_gts, 5, iou_thr, angle_tol_deg),
        "samples": len(all_preds),
    }


def save_checkpoint(path: str | Path, state: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[SiglipGrasp, dict]:
    ckpt = torch.load(path, map_location=device)
    spec = ckpt.get("model_spec")
    if spec is None:
        raise ValueError("Checkpoint has no model_spec; cannot reconstruct the model")
    model = SiglipGrasp({"model": spec}).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt

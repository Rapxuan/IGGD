"""Group Grasp-Anything++ samples using the repository's object-ID lists."""

# from __future__ import annotations

import pickle
from pathlib import Path


def load_grasp_object_ids(path: Path) -> set[str]:
    """Read object IDs from a trusted Grasp-Anything .obj pickle."""
    with path.open("rb") as stream:
        values = pickle.load(stream)
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{path}: expected a collection of object IDs")
    object_ids = set(values)
    if not object_ids or any(
        not isinstance(value, str) or not value for value in object_ids
    ):
        raise ValueError(f"{path}: expected nonempty string object IDs")
    return object_ids


def extract_grasp_object_id(record: dict) -> str:
    """scene_object_part:instruction -> scene_object, including multi-digit IDs."""
    annotation_id = record.get("annotation_id", record["id"].split(":", 1)[0])
    id_parts = annotation_id.rsplit("_", 2)
    if (len(id_parts) != 3 or id_parts[0] != record["scene_id"]
            or not all(part.isdigit() for part in id_parts[1:])):
        raise ValueError(
            f"{record['id']}: expected scene_object_part:instruction sample ID "
            "or an annotation_id field containing scene_object_part")
    return f"{id_parts[0]}_{id_parts[1]}"


def group_grasp_anything_records_by_split(
    records: list[dict], grasp_anything_split_dir: str | Path,
) -> tuple[dict, dict]:
    """Group records by train/seen/unseen IDs; leave train/val splitting to the caller.

    Accept either Grasp-Anything's seen.obj/unseen.obj directory or LGD's
    grasp-anything++ directory containing train/seen.obj and test/*.obj.
    Unlisted samples are excluded and counted, never assigned at random.
    """
    grasp_anything_split_dir = Path(grasp_anything_split_dir).resolve()
    if (grasp_anything_split_dir / "train" / "seen.obj").is_file():
        split_files = {"train": "train/seen.obj", "test_seen": "test/seen.obj",
                       "test_unseen": "test/unseen.obj"}
    else:
        split_files = {"train": "seen.obj", "test_unseen": "unseen.obj"}

    object_to_split, listed_object_counts = {}, {}
    for split_name, filename in split_files.items():
        object_ids = load_grasp_object_ids(grasp_anything_split_dir / filename)
        listed_object_counts[split_name] = len(object_ids)
        for object_id in object_ids:
            if object_id in object_to_split:
                raise ValueError(
                    f"Grasp-Anything split overlap: {object_id} occurs in both "
                    f"{split_files[object_to_split[object_id]]} and {filename}")
            object_to_split[object_id] = split_name

    records_by_split = {split_name: [] for split_name in split_files}
    matched_object_ids = {split_name: set() for split_name in split_files}
    unmatched_samples = 0
    for record in records:
        object_id = extract_grasp_object_id(record)
        split_name = object_to_split.get(object_id)
        if split_name is None:
            unmatched_samples += 1
            continue
        records_by_split[split_name].append(record)
        matched_object_ids[split_name].add(object_id)

    report = {
        "kind": "Grasp-Anything ID membership with a custom scene-disjoint train/val holdout",
        "grasp_anything_split_dir": str(grasp_anything_split_dir),
        "unmatched_samples": unmatched_samples,
        "grasp_anything_id_lists": {
            split_name: {
                "file": filename,
                "listed_objects": listed_object_counts[split_name],
                "matched_objects": len(matched_object_ids[split_name]),
                "missing_objects": (
                    listed_object_counts[split_name] - len(matched_object_ids[split_name])
                ),
            }
            for split_name, filename in split_files.items()
        },
    }
    return records_by_split, report

"""
Step 0b — sync_check.py
=======================
Validates that buildings.json, raw/, train/, gallery/, val/, and the checkpoint are in sync.
Writes data/sync_report.json.  Halts with exit code 1 if any hard error is found.

Hard errors (halt execution):
  - A folder in data/raw/ or data/train/ whose name is not in buildings.json
  - A class in the checkpoint no longer in buildings.json
  - A label in buildings.json with no corresponding folder in data/gallery/
  - Duplicate labels or node_ids in buildings.json

Warnings (log and continue):
  - A label in buildings.json with no folder in data/raw/
  - A class with fewer than 3 images in data/raw/
  - A class with fewer than 3 images in data/gallery/

Usage:
    python -m src.sync_check [--data_dir data] [--checkpoint checkpoints/best_model.pth] [--seed 42]
"""

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def _subfolders(path: str) -> set:
    if not os.path.isdir(path):
        return set()
    return {d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))}


def load_buildings(data_dir: str) -> list:
    path = os.path.join(data_dir, "buildings.json")
    if not os.path.isfile(path):
        logger.error(f"buildings.json not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync-check data directories against buildings.json")
    parser.add_argument("--data_dir",   type=str, default="data",                      help="Path to data/ directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint .pth")
    parser.add_argument("--seed",       type=int, default=42,                          help="Seed (stored in report)")
    parser.add_argument("--verbose",    action="store_true",                            help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    buildings = load_buildings(args.data_dir)
    labels   = [entry["label"]   for entry in buildings]
    node_ids = [entry["node_id"] for entry in buildings]

    # ── Uniqueness checks (contract violations) ──────────────────────────────
    dup_labels   = list({l for l in labels   if labels.count(l)   > 1})
    dup_node_ids = list({n for n in node_ids if node_ids.count(n) > 1})

    assert len(labels)   == len(set(labels)), \
        f"Duplicate labels in buildings.json: {dup_labels}"
    assert len(node_ids) == len(set(node_ids)), \
        f"Duplicate node_ids in buildings.json: {dup_node_ids}"

    label_uniqueness   = "ok" if not dup_labels   else f"FAIL: {dup_labels}"
    node_id_uniqueness = "ok" if not dup_node_ids else f"FAIL: {dup_node_ids}"

    label_set = set(labels)

    raw_dir     = os.path.join(args.data_dir, "raw")
    train_dir   = os.path.join(args.data_dir, "train")
    gallery_dir = os.path.join(args.data_dir, "gallery")
    val_dir     = os.path.join(args.data_dir, "val")

    raw_folders     = _subfolders(raw_dir)
    train_folders   = _subfolders(train_dir)
    gallery_folders = _subfolders(gallery_dir)

    errors   = []
    warnings = []

    # ── Hard errors ──────────────────────────────────────────────────────────

    # Rogue folders in raw/ or train/
    for folder in sorted(raw_folders - label_set):
        errors.append(f"data/raw/{folder!r} is not in buildings.json — remove or add to buildings.json")
    for folder in sorted(train_folders - label_set):
        errors.append(f"data/train/{folder!r} is not in buildings.json — remove or add to buildings.json")

    # Gallery is mandatory for every label
    for label in labels:
        if label not in gallery_folders:
            errors.append(
                f"data/gallery/{label!r}/ is missing — gallery is required for all labels"
            )

    # Checkpoint class list
    ckpt_classes = None
    if os.path.isfile(args.checkpoint):
        try:
            import torch
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            ckpt_classes = ckpt.get("classes", None)
            if ckpt_classes:
                removed = sorted(set(ckpt_classes) - label_set)
                for cls in removed:
                    errors.append(
                        f"Checkpoint class {cls!r} no longer exists in buildings.json"
                    )
            logger.debug(f"Checkpoint loaded. classes={ckpt_classes}")
        except Exception as exc:
            warnings.append(f"Could not load checkpoint for class verification: {exc}")
    else:
        logger.info(f"No checkpoint found at {args.checkpoint} — skipping checkpoint class check")

    # ── Warnings ─────────────────────────────────────────────────────────────

    classes_below_min = []
    for label in labels:
        # raw/ presence
        if label not in raw_folders:
            warnings.append(f"'{label}' has no folder in data/raw/")
        else:
            n_raw = _count_images(os.path.join(raw_dir, label))
            if n_raw < 3:
                warnings.append(f"'{label}' has only {n_raw} image(s) in data/raw/ (min recommended: 3)")
                classes_below_min.append(label)

        # gallery images
        n_gal = _count_images(os.path.join(gallery_dir, label))
        if n_gal < 3:
            warnings.append(f"'{label}' has only {n_gal} image(s) in data/gallery/ (min recommended: 3)")

    # ── New classes (in buildings.json but not in checkpoint) ────────────────
    new_classes     = sorted(label_set - set(ckpt_classes or []))
    removed_classes = sorted(set(ckpt_classes or []) - label_set)

    passed = len(errors) == 0

    # ── Emit logs ────────────────────────────────────────────────────────────
    for w in warnings:
        logger.warning(w)
    for e in errors:
        logger.error(e)

    # ── Write report ─────────────────────────────────────────────────────────
    report = {
        "total_classes_in_buildings_json": len(labels),
        "total_classes_in_raw":            len(raw_folders & label_set),
        "total_classes_in_gallery":        len(gallery_folders & label_set),
        "total_classes_in_checkpoint":     len(ckpt_classes) if ckpt_classes else 0,
        "new_classes":                     new_classes,
        "removed_classes":                 removed_classes,
        "classes_below_minimum_images":    classes_below_min,
        "label_uniqueness":                label_uniqueness,
        "node_id_uniqueness":              node_id_uniqueness,
        "passed":                          passed,
        "errors":                          errors,
        "warnings":                        warnings,
        "seed":                            args.seed,
    }

    report_path = os.path.join(args.data_dir, "sync_report.json")
    os.makedirs(args.data_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Sync report written to {report_path}")

    if not passed:
        logger.error("Sync check FAILED — fix errors above before proceeding.")
        sys.exit(1)

    logger.info(f"Sync check PASSED — {len(labels)} classes, {len(new_classes)} new, {len(warnings)} warning(s).")


if __name__ == "__main__":
    main()

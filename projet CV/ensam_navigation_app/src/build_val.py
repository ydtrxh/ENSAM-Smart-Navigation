"""
Step 0a — build_val.py
======================
Build the fixed validation set from data/gallery/.

Run once when adding new classes. Idempotent — never overwrites existing val/<label>/ folders.

Usage:
    python -m src.build_val [--data_dir data] [--n_images 10] [--seed 42] [--verbose]
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)


def load_buildings(data_dir: str) -> list[dict]:
    path = os.path.join(data_dir, "buildings.json")
    if not os.path.isfile(path):
        logger.error(f"buildings.json not found at: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_val(data_dir: str, n_images: int, seed: int) -> dict:
    set_seed(seed)

    buildings = load_buildings(data_dir)
    labels = [entry["label"] for entry in buildings]

    gallery_dir = os.path.join(data_dir, "gallery")
    val_dir = os.path.join(data_dir, "val")

    stats = {"copied": 0, "skipped_existing": 0, "skipped_no_gallery": 0, "warnings": []}

    for label in labels:
        val_class_dir = os.path.join(val_dir, label)

        # Idempotent: skip if val/<label>/ already exists
        if os.path.isdir(val_class_dir):
            logger.debug(f"[SKIP] '{label}': val/{label}/ already exists — skipping")
            stats["skipped_existing"] += 1
            continue

        gallery_class_dir = os.path.join(gallery_dir, label)
        if not os.path.isdir(gallery_class_dir):
            msg = f"[WARN] '{label}': data/gallery/{label}/ does not exist — skipping"
            logger.warning(msg)
            stats["warnings"].append(msg)
            stats["skipped_no_gallery"] += 1
            continue

        images = [
            f for f in os.listdir(gallery_class_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]

        if len(images) == 0:
            msg = f"[WARN] '{label}': gallery folder exists but contains no images"
            logger.warning(msg)
            stats["warnings"].append(msg)
            continue

        if len(images) < n_images:
            msg = (
                f"[WARN] '{label}': only {len(images)} images in gallery, "
                f"requested {n_images}. Copying all available."
            )
            logger.warning(msg)
            stats["warnings"].append(msg)

        n_copy = min(n_images, len(images))
        selected = random.sample(images, n_copy)
        os.makedirs(val_class_dir, exist_ok=True)

        for img_name in selected:
            src = os.path.join(gallery_class_dir, img_name)
            dst = os.path.join(val_class_dir, img_name)
            shutil.copy2(src, dst)
            logger.debug(f"  Copied {img_name} → val/{label}/")

        logger.info(f"[OK]   '{label}': {n_copy} images copied → val/{label}/")
        stats["copied"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed validation set from gallery images.")
    parser.add_argument("--data_dir",  type=str, default="data", help="Path to data/ directory")
    parser.add_argument("--n_images",  type=int, default=10,     help="Images to copy per class (default: 10)")
    parser.add_argument("--seed",      type=int, default=42,     help="Random seed for reproducibility")
    parser.add_argument("--verbose",   action="store_true",      help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(f"build_val  | data_dir={args.data_dir}  n_images={args.n_images}  seed={args.seed}")
    stats = build_val(args.data_dir, args.n_images, args.seed)

    logger.info(
        f"Done — {stats['copied']} classes populated, "
        f"{stats['skipped_existing']} already existed, "
        f"{stats['skipped_no_gallery']} had no gallery folder."
    )
    if stats["warnings"]:
        logger.warning(f"{len(stats['warnings'])} warning(s) — review above.")


if __name__ == "__main__":
    main()

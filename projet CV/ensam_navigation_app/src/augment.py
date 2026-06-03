"""
Step 1 — augment.py
===================
Offline data augmentation. Reads class list from data/buildings.json.
Writes augmented images to data/train/<label>/.
Writes data/augmentation_report.json.

IMPORTANT: albumentations must be version 1.3.1 (pinned in requirements.txt).
  - CoarseDropout params changed in 1.4+ (max_holes/max_height/max_width renamed)
  - GaussNoise var_limit renamed to std_range in 1.4+
  Both failures are SILENT — wrong params are accepted and ignored.

IMPORTANT: RandomErasing does NOT belong in this file.
  It is a torchvision transform that operates on tensors.
  augment.py works with PIL images via albumentations.
  RandomErasing belongs exclusively in dataset.py after ToTensor().

Usage:
    python -m src.augment [--data_dir data] [--target_count 200] [--seed 42] [--verbose]
    # Development: --target_count 200  (fast iteration)
    # Final run:   --target_count 300  (report quality)
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _check_albumentations_version() -> str:
    try:
        import albumentations as A
        version = A.__version__
        if not version.startswith("1.3"):
            logger.warning(
                f"albumentations {version} detected. Pipeline requires ==1.3.1. "
                "CoarseDropout and GaussNoise APIs changed in 1.4+ causing SILENT data corruption. "
                "Install with: pip install albumentations==1.3.1"
            )
        else:
            logger.info(f"albumentations {version} — OK")
        return version
    except ImportError:
        logger.error("albumentations is not installed. Run: pip install albumentations==1.3.1")
        sys.exit(1)


def _build_pipeline():
    """
    Augmentation pipeline using albumentations==1.3.1.
    All API names are pinned to 1.3.1 conventions.
    DO NOT upgrade without reading the changelog — many renames are SILENT.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2  # noqa: F401  (import check)

    return A.Compose([
        # Group A — Lighting
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.5),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
        A.RandomSunFlare(p=0.2),

        # Group B — Geometry
        A.Perspective(scale=(0.05, 0.15), p=1.0),
        A.Rotate(limit=20, p=0.8),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

        # Group C — Occlusion
        # albumentations==1.3.1: max_holes, max_height, max_width
        # In 1.4+ these were renamed. The failure is SILENT — wrong params are ignored,
        # no dropout is applied, no error is raised. This is why the version is pinned.
        A.CoarseDropout(max_holes=6, max_height=40, max_width=40, p=0.5),

        # Group D — Image quality
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
        # albumentations==1.3.1: var_limit=(low, high)
        # In 1.4+ renamed to std_range. Another SILENT break.
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

        # Group E — Color
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
        A.ToGray(p=0.1),
    ])


def _load_buildings(data_dir: str) -> list:
    path = os.path.join(data_dir, "buildings.json")
    if not os.path.isfile(path):
        logger.error(f"buildings.json not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def augment(data_dir: str, target_count: int, seed: int) -> dict:
    import cv2
    from PIL import Image

    set_seed(seed)
    pipeline = _build_pipeline()

    buildings = _load_buildings(data_dir)
    labels = [entry["label"] for entry in buildings]

    raw_dir   = os.path.join(data_dir, "raw")
    train_dir = os.path.join(data_dir, "train")

    per_class = {}
    classes_processed = 0
    classes_skipped   = 0

    for label in labels:
        raw_class_dir   = os.path.join(raw_dir,   label)
        train_class_dir = os.path.join(train_dir, label)

        # Collect raw images
        if not os.path.isdir(raw_class_dir):
            logger.warning(f"[SKIP] '{label}': no data/raw/{label}/ folder — skipping")
            classes_skipped += 1
            per_class[label] = {"skipped": True, "reason": "no raw folder"}
            continue

        raw_images = [
            f for f in os.listdir(raw_class_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]

        if len(raw_images) < 3:
            logger.warning(
                f"[SKIP] '{label}': only {len(raw_images)} raw image(s) — "
                "minimum 3 required for augmentation"
            )
            classes_skipped += 1
            per_class[label] = {"skipped": True, "reason": f"only {len(raw_images)} raw images"}
            continue

        # Count existing train images
        existing_train = 0
        if os.path.isdir(train_class_dir):
            existing_train = sum(
                1 for f in os.listdir(train_class_dir)
                if os.path.splitext(f)[1].lower() in IMG_EXTS
            )

        if existing_train >= target_count:
            logger.info(
                f"[SKIP] '{label}': already has {existing_train} training images "
                f"(≥ target_count={target_count})"
            )
            classes_skipped += 1
            per_class[label] = {
                "raw": len(raw_images), "augmented": existing_train,
                "added": 0, "skipped": True
            }
            continue

        os.makedirs(train_class_dir, exist_ok=True)

        # Step 1: copy originals to train/
        copied_originals = 0
        for img_name in raw_images:
            src = os.path.join(raw_class_dir, img_name)
            dst = os.path.join(train_class_dir, img_name)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
                copied_originals += 1

        # Step 2: generate augmented images up to target_count
        n_to_generate = max(0, target_count - existing_train - copied_originals)
        generated = 0
        aug_idx   = 0

        while generated < n_to_generate:
            # Pick a random source image
            src_name = random.choice(raw_images)
            src_path = os.path.join(raw_class_dir, src_name)

            # Load via Pillow so accented Windows paths are handled correctly.
            try:
                img_rgb = np.array(Image.open(src_path).convert("RGB"))
            except Exception:
                logger.warning(f"  Could not read {src_path} — skipping")
                continue

            # Augment
            result = pipeline(image=img_rgb)
            aug_img = result["image"]

            # Save
            stem = Path(src_name).stem
            out_name = f"aug_{stem}_{aug_idx:05d}.jpg"
            out_path = os.path.join(train_class_dir, out_name)
            cv2.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

            generated += 1
            aug_idx   += 1

        total_train = existing_train + copied_originals + generated
        logger.info(
            f"[OK]   '{label}': raw={len(raw_images)}  "
            f"copied={copied_originals}  generated={generated}  "
            f"total_train={total_train}"
        )
        per_class[label] = {
            "raw": len(raw_images),
            "augmented": total_train,
            "added": copied_originals + generated,
        }
        classes_processed += 1

    return {
        "classes_processed": classes_processed,
        "classes_skipped":   classes_skipped,
        "per_class":         per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline data augmentation from data/raw/ → data/train/")
    parser.add_argument("--data_dir",      type=str, default="data", help="Path to data/ directory")
    parser.add_argument("--target_count",  type=int, default=200,
                        help="Target images per class (200=dev, 300=final). Default: 200")
    parser.add_argument("--seed",          type=int, default=42,     help="Random seed")
    parser.add_argument("--verbose",       action="store_true",      help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    albu_version = _check_albumentations_version()

    logger.info(
        f"augment.py | data_dir={args.data_dir}  target_count={args.target_count}  seed={args.seed}"
    )

    result = augment(args.data_dir, args.target_count, args.seed)

    report = {
        "albumentations_version": albu_version,
        "target_count":           args.target_count,
        "seed":                   args.seed,
        "classes_processed":      result["classes_processed"],
        "classes_skipped":        result["classes_skipped"],
        "per_class":              result["per_class"],
    }

    report_path = os.path.join(args.data_dir, "augmentation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Augmentation report written to {report_path}")
    logger.info(
        f"Done — {result['classes_processed']} classes augmented, "
        f"{result['classes_skipped']} skipped."
    )


if __name__ == "__main__":
    main()

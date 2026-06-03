"""
Step 5 — evaluate.py
====================
Evaluates the trained model on data/val/.
Loads class list from data/buildings.json and verifies checkpoint classes match exactly.
Raises ClassMismatchError if not.

Outputs:
    outputs/confusion_matrix.png
    outputs/tsne.png
    outputs/eval_report.json   (appended with previous run for comparison)

Usage:
    python -m src.evaluate [--data_dir data] [--checkpoint checkpoints/best_model.pth]
                           [--gallery_path checkpoints/gallery.pkl] [--output_dir outputs]
                           [--seed 42] [--verbose]
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Gallery builder (also used by retrain.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_gallery(model, gallery_dir: str, data_dir: str,
                  device: torch.device, gallery_path: str) -> dict:
    """
    Builds class centroids from data/gallery/.

    Double L2 normalization:
    1. L2-normalize each individual embedding before averaging.
    2. L2-normalize the resulting centroid.

    WHY DOUBLE NORMALIZATION:
    Averaging unnormalized embeddings biases the centroid toward high-magnitude
    outliers. Averaging pre-normalized embeddings gives equal weight to every image.
    The final normalization places the centroid on the unit hypersphere where
    cosine similarity is well-defined.

    NEVER fall back to data/train/ — training images contaminate the gallery
    and inflate Recall@1 artificially.
    """
    from src.dataset import GalleryDataset

    dataset = GalleryDataset(gallery_dir, data_dir)
    if len(dataset) == 0:
        logger.warning(f"Gallery dataset is empty at {gallery_dir}")
        return {}

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()

    class_embeddings: dict = {}
    for images, labels in loader:
        images = images.to(device)
        embs   = model(images)  # already L2-normalized from model.forward()
        for emb, label in zip(embs, labels):
            if label not in class_embeddings:
                class_embeddings[label] = []
            class_embeddings[label].append(emb.cpu())

    centroids = {}
    for label, embs in class_embeddings.items():
        normed   = [F.normalize(e.unsqueeze(0), dim=-1).squeeze(0) for e in embs]
        centroid = torch.stack(normed).mean(dim=0)
        centroids[label] = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0)

    os.makedirs(os.path.dirname(gallery_path) or ".", exist_ok=True)
    with open(gallery_path, "wb") as f:
        pickle.dump(centroids, f)

    logger.info(f"Gallery rebuilt: {len(centroids)} centroids → {gallery_path}")
    model.train()
    return centroids


# ─────────────────────────────────────────────────────────────────────────────
# Threshold estimation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_threshold(model, centroids: dict, val_dir: str,
                        data_dir: str, device: torch.device) -> tuple:
    from src.dataset import GalleryDataset

    dataset = GalleryDataset(val_dir, data_dir)
    if len(dataset) == 0:
        logger.warning("Val set empty — using default threshold 0.6")
        return 0.6, 0.0, 0

    loader          = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    gallery_classes = sorted(centroids.keys())
    centroid_tensor = torch.stack([centroids[c] for c in gallery_classes]).to(device)

    model.eval()
    sims, correct_flags = [], []

    for images, labels in loader:
        images  = images.to(device)
        embs    = model(images)
        sim_m   = torch.matmul(embs, centroid_tensor.t())
        max_sim, max_idx = sim_m.max(dim=1)

        for score, idx, label in zip(max_sim, max_idx, labels):
            sims.append(score.item())
            correct_flags.append(1 if gallery_classes[idx.item()] == label else 0)

    sims          = np.array(sims)
    correct_flags = np.array(correct_flags)
    n             = len(sims)

    best_t  = 0.6
    best_f1 = 0.0
    for t in np.linspace(0.0, 1.0, 100):
        preds = (sims >= t).astype(int)
        try:
            f1 = f1_score(correct_flags, preds, zero_division=0)
        except Exception:
            f1 = 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)

    logger.info(f"Unknown threshold: {best_t:.4f} (F1={best_f1:.4f}, n={n} val images)")
    model.train()
    return best_t, best_f1, n


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader: DataLoader, classes: list,
             device: torch.device) -> dict:
    """
    Evaluate on the val set using leave-one-out retrieval.
    Returns dict with all metrics.
    """
    model.eval()

    all_embeddings = []
    all_labels_str = []

    for images, labels in val_loader:
        images = images.to(device)
        embs   = model(images)
        all_embeddings.append(embs.cpu())
        all_labels_str.extend(labels)

    if not all_embeddings:
        logger.error("No validation images found")
        return {}

    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = F.normalize(embeddings, dim=1)
    N          = embeddings.size(0)

    # Integer label tensor
    class_to_idx = {c: i for i, c in enumerate(classes)}
    try:
        label_ids = torch.tensor([class_to_idx[l] for l in all_labels_str])
    except KeyError as e:
        logger.error(f"Label {e} not in classes from buildings.json — run sync_check.py")
        sys.exit(1)

    # ── Cosine similarity ─────────────────────────────────────────────────
    sim = torch.matmul(embeddings, embeddings.t())
    sim.fill_diagonal_(-1e9)

    max_k         = min(5, N - 1)
    top_k_indices = sim.topk(max_k, dim=1).indices

    def recall_at(k: int) -> float:
        hits = (label_ids.unsqueeze(1) == label_ids[top_k_indices[:, :k]]).any(dim=1)
        return hits.float().mean().item()

    r1 = recall_at(1)
    r3 = recall_at(min(3, max_k))
    r5 = recall_at(max_k)

    # ── Top-1 predictions for classification report ────────────────────────
    top1_idx   = top_k_indices[:, 0]
    preds      = label_ids[top1_idx].numpy()
    targets    = label_ids.numpy()

    # ── Classification report ─────────────────────────────────────────────
    present_classes = sorted(set(targets.tolist()))
    present_names   = [classes[i] for i in present_classes]

    report_dict = classification_report(
        targets, preds,
        labels=present_classes,
        target_names=present_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        targets, preds,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
    )
    logger.info("\n" + report_str)

    macro_f1 = report_dict.get("macro avg", {}).get("f1-score", 0.0)

    # ── Intra / inter class distances ─────────────────────────────────────
    dist = 1.0 - sim
    dist.fill_diagonal_(1e9)

    same_class = label_ids.unsqueeze(0) == label_ids.unsqueeze(1)
    diag_mask  = torch.eye(N, dtype=torch.bool)
    same_class = same_class & ~diag_mask
    diff_class = ~same_class & ~diag_mask

    intra = dist[same_class].mean().item() if same_class.any() else float("nan")
    inter = dist[diff_class].mean().item() if diff_class.any() else float("nan")

    model.train()
    return {
        "recall_at_1":               round(r1,    4),
        "recall_at_3":               round(r3,    4),
        "recall_at_5":               round(r5,    4),
        "f1":                        round(macro_f1, 4),
        "mean_intra_class_distance": round(intra, 4),
        "mean_inter_class_distance": round(inter, 4),
        "n_val_images":              N,
        "embeddings":                embeddings,
        "label_ids":                 label_ids,
        "preds":                     preds,
        "targets":                   targets,
        "class_names":               present_names,
        "classification_report":     report_dict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(targets, preds, class_names: list, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(targets, preds)
    n  = len(class_names)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), max(8, n * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix — Validation Set",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────────────────────────────────────

def save_tsne(embeddings: torch.Tensor, label_ids: torch.Tensor,
              classes: list, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n_samples = embeddings.size(0)
    perp      = min(15, max(5, n_samples // 5))  # cap perplexity to avoid sklearn error

    logger.info(f"Running t-SNE (n={n_samples}, perplexity={perp}) …")
    tsne   = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings.numpy())   # (N, 2)

    unique_ids = sorted(set(label_ids.tolist()))
    cmap       = plt.colormaps.get_cmap("tab20")
    colors     = {uid: cmap(i / max(len(unique_ids) - 1, 1)) for i, uid in enumerate(unique_ids)}

    fig, ax = plt.subplots(figsize=(12, 9))
    for uid in unique_ids:
        mask = label_ids == uid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=colors[uid],
            label=classes[uid] if uid < len(classes) else str(uid),
            alpha=0.7, s=20,
        )

    ax.set_title(f"t-SNE Embedding Visualization (perplexity={perp})")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, markerscale=2)
    plt.tight_layout()

    path = os.path.join(output_dir, "tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Eval report (append comparison)
# ─────────────────────────────────────────────────────────────────────────────

def save_eval_report(current: dict, checkpoint_meta: dict, output_dir: str) -> None:
    report_path = os.path.join(output_dir, "eval_report.json")

    previous = None
    if os.path.isfile(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            previous = existing.get("current", None)
        except Exception:
            pass

    def _fmt_delta(cur, prev, key):
        if prev is None or key not in prev:
            return "N/A"
        d = cur.get(key, 0) - prev.get(key, 0)
        return f"{d:+.3f}"

    delta = None
    if previous:
        delta_keys = ["recall_at_1", "f1", "mean_intra_class_distance", "mean_inter_class_distance"]
        delta = {k: _fmt_delta(current, previous, k) for k in delta_keys}

    report = {
        "current":  {**current, **checkpoint_meta},
        "previous": previous,
        "delta":    delta,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Eval report written to {report_path}")
    if delta:
        logger.info(
            f"Delta vs previous run — "
            f"R@1: {delta['recall_at_1']}  "
            f"F1: {delta['f1']}  "
            f"Intra: {delta['mean_intra_class_distance']}  "
            f"Inter: {delta['mean_inter_class_distance']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the campus building recognition model on val set")
    parser.add_argument("--data_dir",      type=str, default="data",                       help="data/ directory")
    parser.add_argument("--checkpoint",    type=str, default="checkpoints/best_model.pth",  help="Checkpoint path")
    parser.add_argument("--gallery_path",  type=str, default="checkpoints/gallery.pkl",     help="Gallery .pkl path")
    parser.add_argument("--output_dir",    type=str, default="outputs",                     help="Outputs directory")
    parser.add_argument("--rebuild_gallery", action="store_true",                           help="Rebuild gallery before eval")
    parser.add_argument("--seed",          type=int, default=42,                            help="Random seed")
    parser.add_argument("--verbose",       action="store_true",                             help="DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Imports ───────────────────────────────────────────────────────────
    from src.dataset       import load_buildings, GalleryDataset, ClassMismatchError, get_val_transforms
    from cv_engine.model   import MetricLearningModel

    # ── Load buildings.json ───────────────────────────────────────────────
    buildings = load_buildings(args.data_dir)
    classes   = [b["label"]   for b in buildings]
    node_ids  = {b["label"]: b["node_id"] for b in buildings}
    logger.info(f"buildings.json: {len(classes)} classes")

    # ── Load checkpoint and verify class list ─────────────────────────────
    if not os.path.isfile(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)

    ckpt_classes = ckpt.get("classes", [])
    if sorted(ckpt_classes) != sorted(classes):
        missing_in_ckpt = sorted(set(classes)       - set(ckpt_classes))
        extra_in_ckpt   = sorted(set(ckpt_classes)  - set(classes))
        raise ClassMismatchError(
            f"Checkpoint classes do not match buildings.json!\n"
            f"  In buildings.json but not checkpoint: {missing_in_ckpt}\n"
            f"  In checkpoint but not buildings.json: {extra_in_ckpt}\n"
            "Run sync_check.py and retrain."
        )
    logger.info("Class lists match ✓")

    # ── Model ─────────────────────────────────────────────────────────────
    model = MetricLearningModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    threshold    = ckpt.get("threshold", 0.6)
    loss_fn_name = ckpt.get("loss_function", "triplet")
    seed_ckpt    = ckpt.get("seed", args.seed)

    # ── Optionally rebuild gallery ─────────────────────────────────────────
    gallery_dir = os.path.join(args.data_dir, "gallery")
    if args.rebuild_gallery or not os.path.isfile(args.gallery_path):
        logger.info("Building reference gallery …")
        centroids = build_gallery(model, gallery_dir, args.data_dir, device, args.gallery_path)
        if centroids:
            new_t, f1_t, n_t = estimate_threshold(
                model, centroids, os.path.join(args.data_dir, "val"),
                args.data_dir, device
            )
            threshold = new_t
            ckpt["threshold"] = threshold
            torch.save(ckpt, args.checkpoint)
            logger.info(f"Updated checkpoint threshold → {threshold:.4f}")

    # ── Validation DataLoader ─────────────────────────────────────────────
    val_dir     = os.path.join(args.data_dir, "val")
    val_dataset = GalleryDataset(val_dir, args.data_dir, transform=get_val_transforms())
    if len(val_dataset) == 0:
        logger.error(f"No val images found in {val_dir}. Run build_val.py first.")
        sys.exit(1)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    logger.info(f"Val set: {len(val_dataset)} images")

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = evaluate(model, val_loader, classes, device)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Recall@1  : {results['recall_at_1']:.4f}")
    print(f"Recall@3  : {results['recall_at_3']:.4f}")
    print(f"Recall@5  : {results['recall_at_5']:.4f}")
    print(f"Macro F1  : {results['f1']:.4f}")
    print(f"Intra dist: {results['mean_intra_class_distance']:.4f}  (target < 0.30)")
    print(f"Inter dist: {results['mean_inter_class_distance']:.4f}  (target > 0.65)")
    print(f"Threshold : {threshold:.4f}")
    print(f"Val images: {results['n_val_images']}")
    print("=" * 50 + "\n")

    if results["mean_intra_class_distance"] > 0.3 or results["mean_inter_class_distance"] < 0.65:
        logger.warning(
            "Embedding quality below target. "
            "Intra > 0.3 or Inter < 0.65 may indicate partial collapse. "
            "Consider more training epochs or additional data."
        )

    # ── Confusion matrix ──────────────────────────────────────────────────
    save_confusion_matrix(
        results["targets"], results["preds"],
        results["class_names"], args.output_dir
    )

    # ── t-SNE ─────────────────────────────────────────────────────────────
    save_tsne(
        results["embeddings"], results["label_ids"],
        classes, args.output_dir
    )

    # ── Eval report ───────────────────────────────────────────────────────
    current_metrics = {
        "recall_at_1":               results["recall_at_1"],
        "recall_at_3":               results["recall_at_3"],
        "recall_at_5":               results["recall_at_5"],
        "f1":                        results["f1"],
        "mean_intra_class_distance": results["mean_intra_class_distance"],
        "mean_inter_class_distance": results["mean_inter_class_distance"],
        "n_val_images":              results["n_val_images"],
        "threshold":                 round(threshold, 4),
    }
    checkpoint_meta = {
        "loss_function": loss_fn_name,
        "seed":          seed_ckpt,
        "epoch":         ckpt.get("epoch", "?"),
    }
    save_eval_report(current_metrics, checkpoint_meta, args.output_dir)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

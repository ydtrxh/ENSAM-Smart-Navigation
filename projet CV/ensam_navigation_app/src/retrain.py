"""
Step 3 — retrain.py
===================
Full retraining / fine-tuning script for the campus building recognition model.

Scenario A — New classes added or --from_scratch:
    Fresh EfficientNet-B0 with ImageNet weights, blocks 0-5 frozen, train from epoch 0.

Scenario B — Same classes, more data:
    Load checkpoint weights + optimizer state, resume from last epoch.
    All LRs multiplied by 0.1 (standard fine-tune reduction — 0.5 risks catastrophic forgetting).

Usage:
    # Scenario A (new classes):
    python -m src.retrain --data_dir data --loss triplet --seed 42

    # Scenario B (fine-tune with more data):
    python -m src.retrain --data_dir data --checkpoint checkpoints/best_model.pth --loss triplet

    # Compare loss functions (same seed → fair comparison):
    python -m src.retrain --loss triplet  --seed 42
    python -m src.retrain --loss arcface  --seed 42
    python -m src.retrain --loss combined --seed 42
"""

import argparse
import datetime
import json
import logging
import os
import pickle
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Makes convolutions deterministic at a small speed cost.
    # Required for reproducible training across runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Imports from the pipeline (relative to ensam_navigation_app/)
# ─────────────────────────────────────────────────────────────────────────────

def _import_pipeline():
    from src.dataset import BuildingsDataset, GalleryDataset, PKSampler
    from src.dataset import get_train_transforms, get_val_transforms
    from src.losses  import build_loss
    from cv_engine.model import MetricLearningModel
    return BuildingsDataset, GalleryDataset, PKSampler, get_train_transforms, get_val_transforms, build_loss, MetricLearningModel


# ─────────────────────────────────────────────────────────────────────────────
# Validation metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_val_metrics(model: nn.Module, val_loader: DataLoader, device: torch.device, loss_fn=None) -> dict:
    """
    Evaluates the model on the validation set.
    Uses the val set itself as both queries and gallery (leave-one-out style).

    Returns: dict with val_loss, recall@1/3/5, intra_dist, inter_dist
    """
    model.eval()

    all_embeddings = []
    all_labels     = []
    total_loss     = 0.0
    n_batches      = 0

    for images, labels in val_loader:
        images = images.to(device)
        emb    = model(images)
        all_embeddings.append(emb.cpu())

        if isinstance(labels[0], str):
            # String labels from GalleryDataset — skip loss
            all_labels.extend(labels)
        else:
            labels_t = labels.to(device)
            all_labels.extend(labels.tolist())
            if loss_fn is not None:
                try:
                    loss, _, _ = loss_fn(emb, labels_t)
                    total_loss += loss.item()
                    n_batches  += 1
                except Exception:
                    pass

    embeddings = torch.cat(all_embeddings, dim=0)   # (N, 128)
    embeddings = F.normalize(embeddings, dim=1)

    # Convert string labels to integer indices for recall computation
    if isinstance(all_labels[0], str):
        unique_labels = sorted(set(all_labels))
        label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
        label_ids     = torch.tensor([label_to_idx[l] for l in all_labels])
    else:
        label_ids = torch.tensor(all_labels)

    N = embeddings.size(0)

    # ── Cosine similarity matrix ──────────────────────────────────────────
    sim = torch.matmul(embeddings, embeddings.t())   # (N, N)

    # Exclude self-similarity (set diagonal to -inf)
    sim.fill_diagonal_(-1e9)

    # ── Recall @ K ────────────────────────────────────────────────────────
    max_k = min(5, N - 1)
    top_k_indices = sim.topk(max_k, dim=1).indices   # (N, max_k)

    def recall_at(k: int) -> float:
        hits = (label_ids.unsqueeze(1) == label_ids[top_k_indices[:, :k]]).any(dim=1)
        return hits.float().mean().item()

    r1 = recall_at(1)
    r3 = recall_at(min(3, max_k))
    r5 = recall_at(max_k)

    # ── Intra / inter class distances ─────────────────────────────────────
    dist = 1.0 - sim
    dist.fill_diagonal_(1e9)  # exclude self

    same_class = label_ids.unsqueeze(0) == label_ids.unsqueeze(1)
    diff_class = ~same_class

    diag = torch.eye(N, dtype=torch.bool)
    same_class = same_class & ~diag

    intra = dist[same_class].mean().item() if same_class.any() else float("nan")
    inter = dist[diff_class].mean().item() if diff_class.any() else float("nan")

    val_loss = total_loss / n_batches if n_batches > 0 else float("nan")

    model.train()
    return {
        "val_loss": val_loss,
        "recall_at_1": r1,
        "recall_at_3": r3,
        "recall_at_5": r5,
        "mean_intra_class_distance": intra,
        "mean_inter_class_distance": inter,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gallery builder (Step 4 — called at end of training for smoke test)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_gallery(model: nn.Module, gallery_dir: str, data_dir: str,
                  device: torch.device, gallery_path: str) -> dict:
    """
    Compute per-class centroids from data/gallery/ using double L2 normalization.

    WHY DOUBLE NORMALIZATION:
    Averaging unnormalized embeddings then normalizing biases toward high-magnitude
    outliers. Averaging pre-normalized embeddings gives equal weight to every image.
    The final normalization places the centroid on the unit hypersphere.

    NEVER fall back to data/train/ — that contaminates gallery with augmented images
    and artificially inflates Recall@1.
    """
    from src.dataset import GalleryDataset

    dataset = GalleryDataset(gallery_dir, data_dir)
    if len(dataset) == 0:
        logger.warning("Gallery dataset is empty — cannot build gallery centroids")
        return {}

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()

    class_embeddings: dict = {}
    for images, labels in loader:
        images = images.to(device)
        embs   = model(images)                           # already L2-normalized by model
        for emb, label in zip(embs, labels):
            if label not in class_embeddings:
                class_embeddings[label] = []
            class_embeddings[label].append(emb.cpu())

    centroids = {}
    for label, embs in class_embeddings.items():
        # Step 1: L2-normalize each individual embedding
        normed = [F.normalize(e.unsqueeze(0), dim=-1).squeeze(0) for e in embs]
        # Step 2: Average normalized embeddings
        centroid = torch.stack(normed).mean(dim=0)
        # Step 3: L2-normalize the centroid itself
        centroids[label] = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0)

    os.makedirs(os.path.dirname(gallery_path) or ".", exist_ok=True)
    with open(gallery_path, "wb") as f:
        pickle.dump(centroids, f)

    logger.info(f"Gallery saved to {gallery_path} ({len(centroids)} classes)")
    model.train()
    return centroids


# ─────────────────────────────────────────────────────────────────────────────
# Unknown threshold estimation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_threshold(model: nn.Module, centroids: dict, val_dir: str,
                        data_dir: str, device: torch.device) -> tuple:
    """
    Estimate optimal cosine similarity threshold on data/val/.
    Maximizes F1 over 100 candidates in [0.0, 1.0].
    """
    from src.dataset import GalleryDataset
    from sklearn.metrics import f1_score

    dataset = GalleryDataset(val_dir, data_dir)
    if len(dataset) == 0:
        logger.warning("Val set is empty — cannot estimate threshold")
        return 0.6, 0.0, 0

    loader   = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    gallery_classes = sorted(centroids.keys())
    centroid_tensor = torch.stack([centroids[c] for c in gallery_classes]).to(device)

    model.eval()
    sims, true_flags = [], []   # similarity scores and whether top match is correct

    for images, labels in loader:
        images = images.to(device)
        embs   = model(images)
        sim_m  = torch.matmul(embs, centroid_tensor.t())
        max_sim, max_idx = sim_m.max(dim=1)

        for score, idx, label in zip(max_sim, max_idx, labels):
            sims.append(score.item())
            true_flags.append(1 if gallery_classes[idx.item()] == label else 0)

    sims       = np.array(sims)
    true_flags = np.array(true_flags)
    n          = len(sims)

    best_t  = 0.6
    best_f1 = 0.0

    for t in np.linspace(0.0, 1.0, 100):
        # Predict 1 (known) if score >= t, else 0 (unknown)
        preds = (sims >= t).astype(int)
        try:
            f1 = f1_score(true_flags, preds, zero_division=0)
        except Exception:
            f1 = 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    logger.info(f"Unknown threshold: {best_t:.4f} (F1={best_f1:.4f}, n={n} val images)")
    model.train()
    return float(best_t), float(best_f1), n


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(checkpoint_path: str, gallery_path: str, data_dir: str, node_id_set: set) -> None:
    """
    Integration smoke test: verifies every CV prediction maps to a valid node_id.
    Logs WARNING (does not halt) if any known-class image is classified as 'unknown'.
    """
    try:
        from cv_engine.inference import predict
    except ImportError:
        logger.warning("Could not import cv_engine.inference — skipping smoke test")
        return

    import json
    buildings = []
    with open(os.path.join(data_dir, "buildings.json"), "r", encoding="utf-8") as f:
        buildings = json.load(f)

    labels     = [b["label"] for b in buildings]
    train_dir  = os.path.join(data_dir, "train")

    valid_count   = 0
    unknown_count = 0
    tested        = 0

    for label in labels:
        class_dir = os.path.join(train_dir, label)
        if not os.path.isdir(class_dir):
            continue
        images = [
            os.path.join(class_dir, f) for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not images:
            continue
        sample = random.choice(images)
        try:
            pred_class, score = predict(sample, gallery_path, checkpoint_path)
        except Exception as exc:
            logger.warning(f"Smoke test predict failed for '{label}': {exc}")
            continue

        tested += 1
        if pred_class == "unknown":
            logger.warning(f"  [SMOKE] '{label}' predicted as 'unknown' (score={score:.3f})")
            unknown_count += 1
        else:
            # Look up node_id for predicted class
            pred_node = next((b["node_id"] for b in buildings if b["label"] == pred_class), None)
            if pred_node and pred_node in node_id_set:
                valid_count += 1
                logger.debug(f"  [SMOKE] '{label}' → '{pred_class}' → node_id='{pred_node}' ✓")
            else:
                logger.warning(f"  [SMOKE] '{label}' → '{pred_class}' → node_id='{pred_node}' NOT in buildings.json")

    print(f"\nIntegration smoke test : {valid_count}/{tested} predictions returned valid node_ids")
    print(f"Flagged as unknown     : {unknown_count}/{tested}")
    print("Ready for navigation pipeline." if unknown_count == 0 else
          "WARNING: some images flagged as unknown — consider retraining or lowering threshold.")


# ─────────────────────────────────────────────────────────────────────────────
# Curves
# ─────────────────────────────────────────────────────────────────────────────

def save_curves(stats_list: list, output_dir: str) -> None:
    """Save retrain_curves.png and embedding_quality_curves.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs      = [s["epoch"]       for s in stats_list]
    train_loss  = [s["train_loss"]  for s in stats_list]
    val_loss    = [s.get("val_loss", float("nan")) for s in stats_list]
    r1          = [s["recall_at_1"] for s in stats_list]
    r3          = [s["recall_at_3"] for s in stats_list]
    r5          = [s["recall_at_5"] for s in stats_list]
    intra       = [s["mean_intra_class_distance"] for s in stats_list]
    inter       = [s["mean_inter_class_distance"] for s in stats_list]

    # ── Recall & Loss curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", color="#e74c3c")
    ax.plot(epochs, val_loss,   label="Val Loss",   color="#e67e22", linestyle="--")
    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, r1, label="R@1", color="#2ecc71", linewidth=2)
    ax.plot(epochs, r3, label="R@3", color="#3498db", linestyle="--")
    ax.plot(epochs, r5, label="R@5", color="#9b59b6", linestyle=":")
    ax.set_title("Validation Recall@K")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Recall")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "retrain_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

    # ── Embedding quality curves ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, intra, label="Intra-class dist", color="#e74c3c", linewidth=2)
    ax.plot(epochs, inter, label="Inter-class dist", color="#2ecc71", linewidth=2)
    ax.axhline(0.3, color="#e74c3c", linestyle=":", alpha=0.5, label="Intra threshold (0.3)")
    ax.axhline(0.6, color="#2ecc71", linestyle=":", alpha=0.5, label="Inter threshold (0.6)")
    ax.set_title("Embedding Quality (Intra vs Inter class cosine distance)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Cosine Distance")
    ax.legend(); ax.grid(alpha=0.3)

    path = os.path.join(output_dir, "embedding_quality_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain / fine-tune the campus building recognition model")
    parser.add_argument("--data_dir",        type=str,   default="data",                       help="data/ directory")
    parser.add_argument("--checkpoint",      type=str,   default="checkpoints/best_model.pth",  help="Checkpoint path")
    parser.add_argument("--gallery_path",    type=str,   default="checkpoints/gallery.pkl",     help="Gallery .pkl path")
    parser.add_argument("--output_dir",      type=str,   default="outputs",                     help="Outputs directory")
    parser.add_argument("--loss",            type=str,   default="triplet",                     choices=["triplet", "arcface", "combined"])
    parser.add_argument("--from_scratch",    action="store_true",                               help="Force Scenario A")
    parser.add_argument("--seed",            type=int,   default=42,                            help="Random seed")
    parser.add_argument("--epochs",          type=int,   default=50,                            help="Max epochs")
    parser.add_argument("--patience",        type=int,   default=7,                             help="Early stopping patience")
    parser.add_argument("--batch_p",         type=int,   default=8,                             help="P classes per batch")
    parser.add_argument("--batch_k",         type=int,   default=4,                             help="K images per class")
    parser.add_argument("--margin",          type=float, default=0.3,                           help="Triplet margin")
    parser.add_argument("--arcface_margin",  type=float, default=0.5,                           help="ArcFace angular margin")
    parser.add_argument("--arcface_scale",   type=float, default=64.0,                          help="ArcFace scale")
    parser.add_argument("--arcface_weight",  type=float, default=0.5,                           help="ArcFace weight in combined loss")
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--no_smoke_test",   action="store_true",                               help="Skip smoke test")
    parser.add_argument("--verbose",         action="store_true",                               help="DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Seed ─────────────────────────────────────────────────────────────
    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    # ── Imports ───────────────────────────────────────────────────────────
    (BuildingsDataset, GalleryDataset, PKSampler,
     get_train_transforms, get_val_transforms,
     build_loss, MetricLearningModel) = _import_pipeline()

    from src.dataset import load_buildings, ClassMismatchError

    # ── Directories ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir,                   exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    # ── Load buildings.json — single source of truth ──────────────────────
    buildings  = load_buildings(args.data_dir)
    classes    = [b["label"]   for b in buildings]
    node_ids   = [b["node_id"] for b in buildings]
    node_id_set = set(node_ids)
    scale_factor = buildings[0].get("scale_factor", 1.0) if buildings else 1.0
    num_classes = len(classes)

    logger.info(f"Buildings loaded: {num_classes} classes from buildings.json")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Detect scenario ───────────────────────────────────────────────────
    scenario         = "A"
    start_epoch      = 0
    previous_best_r1 = 0.0
    optimizer_state  = None
    ckpt_lr_mult     = 1.0

    if not args.from_scratch and os.path.isfile(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            ckpt_classes = ckpt.get("classes", [])
            if set(ckpt_classes) == set(classes) and sorted(ckpt_classes) == sorted(classes):
                scenario        = "B"
                start_epoch     = ckpt.get("epoch", 0)
                previous_best_r1 = ckpt.get("best_recall_at_1", 0.0)
                optimizer_state = ckpt.get("optimizer_state_dict", None)
                ckpt_lr_mult    = 0.1   # LR × 0.1 for fine-tuning
                logger.info(
                    f"Scenario B detected — fine-tuning from epoch {start_epoch}, "
                    f"LR × {ckpt_lr_mult:.1f}, previous R@1={previous_best_r1:.4f}"
                )
            else:
                new_cls = sorted(set(classes) - set(ckpt_classes))
                logger.info(
                    f"Scenario A — class list changed. New classes: {new_cls}. "
                    "Starting from scratch."
                )
        except Exception as exc:
            logger.warning(f"Could not read checkpoint ({exc}) — defaulting to Scenario A")
    else:
        logger.info("Scenario A — no checkpoint or --from_scratch specified")

    # ── Backup checkpoint before overwriting ──────────────────────────────
    if os.path.isfile(args.checkpoint):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.checkpoint.replace(".pth", f"_{ts}.pth")
        shutil.copy(args.checkpoint, backup_path)
        logger.info(f"Checkpoint backed up to {backup_path}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = MetricLearningModel().to(device)

    if scenario == "B":
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Strict=False in case architecture changed slightly
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        logger.info(f"Loaded checkpoint weights (missing={missing}, unexpected={unexpected})")

    # ── Loss function ─────────────────────────────────────────────────────
    loss_fn = build_loss(
        loss_name      = args.loss,
        num_classes    = num_classes,
        margin         = args.margin,
        arcface_margin = args.arcface_margin,
        arcface_scale  = args.arcface_scale,
        arcface_weight = args.arcface_weight,
        device         = device,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    lr_backbone = (1e-5 if scenario == "A" else 1e-5 * ckpt_lr_mult)
    lr_head     = (1e-4 if scenario == "A" else 1e-4 * ckpt_lr_mult)

    param_groups = [
        {"params": model.backbone.features[6:].parameters(), "lr": lr_backbone, "name": "backbone_6_7"},
        {"params": model.projection_head.parameters(),       "lr": lr_head,     "name": "proj_head"},
    ]
    # Include loss_fn parameters (e.g. ArcFace weight matrix) if any
    loss_params = list(loss_fn.parameters())
    if loss_params:
        param_groups.append({"params": loss_params, "lr": lr_head, "name": "loss_fn"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    if scenario == "B" and optimizer_state:
        try:
            optimizer.load_state_dict(optimizer_state)
            # Apply LR scale to all groups
            for g in optimizer.param_groups:
                g["lr"] *= ckpt_lr_mult
            logger.info("Optimizer state restored and LRs scaled.")
        except Exception as exc:
            logger.warning(f"Could not restore optimizer state: {exc} — using fresh optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    # ── Data loaders ──────────────────────────────────────────────────────
    train_dataset = BuildingsDataset(args.data_dir, transform=get_train_transforms())
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Run augment.py first.")
        sys.exit(1)

    sampler = PKSampler(train_dataset.targets, p=args.batch_p, k=args.batch_k)
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    val_dir = os.path.join(args.data_dir, "val")
    val_dataset = GalleryDataset(val_dir, args.data_dir, transform=get_val_transforms())
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    logger.info(
        f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images | "
        f"Batches/epoch: {len(train_loader)}"
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_r1        = previous_best_r1
    best_epoch     = start_epoch
    patience_count = 0
    stats_list     = []

    stats_path = os.path.join(args.output_dir, "training_stats.json")
    # Load existing stats if resuming
    if scenario == "B" and os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            stats_list = existing.get("epochs", [])
            logger.info(f"Loaded {len(stats_list)} existing epoch stats from {stats_path}")
        except Exception:
            stats_list = []

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total params    : {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss   = 0.0
        epoch_active = 0.0
        epoch_hard   = 0.0
        n_batches    = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(images)
            loss, active_pct, hard_neg_pct = loss_fn(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss   += loss.item()
            epoch_active += active_pct
            epoch_hard   += hard_neg_pct
            n_batches    += 1

        scheduler.step()

        train_loss  = epoch_loss   / max(n_batches, 1)
        active_pct  = epoch_active / max(n_batches, 1) * 100
        hard_neg_p  = epoch_hard   / max(n_batches, 1) * 100

        # Validation
        val_metrics = compute_val_metrics(model, val_loader, device, loss_fn)
        r1 = val_metrics["recall_at_1"]
        r3 = val_metrics["recall_at_3"]
        r5 = val_metrics["recall_at_5"]
        intra = val_metrics["mean_intra_class_distance"]
        inter = val_metrics["mean_inter_class_distance"]
        val_loss = val_metrics["val_loss"]

        # Per-epoch log
        logger.info(
            f"Epoch {epoch+1:3d}/{start_epoch + args.epochs} | "
            f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | "
            f"Active Triplets: {active_pct:.0f}% | Hard Neg: {hard_neg_p:.0f}% | "
            f"R@1: {r1:.3f} | R@3: {r3:.3f} | R@5: {r5:.3f} | "
            f"Intra: {intra:.2f} | Inter: {inter:.2f}"
        )

        epoch_stats = {
            "epoch":                       epoch + 1,
            "train_loss":                  round(train_loss, 4),
            "val_loss":                    round(val_loss, 4),
            "active_triplets_pct":         round(active_pct, 1),
            "hard_negatives_pct":          round(hard_neg_p, 1),
            "recall_at_1":                 round(r1, 4),
            "recall_at_3":                 round(r3, 4),
            "recall_at_5":                 round(r5, 4),
            "mean_intra_class_distance":   round(intra, 4),
            "mean_inter_class_distance":   round(inter, 4),
        }
        stats_list.append(epoch_stats)

        # Save training stats (append each epoch, not overwritten)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "loss_function": args.loss,
                "seed":          args.seed,
                "epochs":        stats_list,
            }, f, ensure_ascii=False, indent=2)

        # Early stopping + checkpoint saving
        if r1 > best_r1:
            best_r1    = r1
            best_epoch = epoch + 1
            patience_count = 0

            import sys as _sys
            ckpt_payload = {
                "model_state_dict":      model.state_dict(),
                "optimizer_state_dict":  optimizer.state_dict(),
                "epoch":                 epoch + 1,
                "best_recall_at_1":      best_r1,
                "classes":               classes,
                "threshold":             0.6,   # updated by estimate_threshold()
                "scale_factor":          scale_factor,
                "scenario":              scenario,
                "loss_function":         args.loss,
                "seed":                  args.seed,
                "albumentations_version": "1.3.1",
                "torch_version":         torch.__version__,
                "torchvision_version":   torchvision.__version__,
                "python_version":        _sys.version,
            }
            torch.save(ckpt_payload, args.checkpoint)
            logger.info(f"  → Checkpoint saved (R@1={best_r1:.4f}, epoch={epoch+1})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.info(
                    f"Early stopping: no improvement for {args.patience} epochs "
                    f"(best R@1={best_r1:.4f} at epoch {best_epoch})"
                )
                break

    # ── Curves ────────────────────────────────────────────────────────────
    if stats_list:
        save_curves(stats_list, args.output_dir)

    # ── Rebuild gallery + threshold ───────────────────────────────────────
    gallery_dir = os.path.join(args.data_dir, "gallery")
    centroids   = build_gallery(model, gallery_dir, args.data_dir, device, args.gallery_path)

    threshold, f1_val, n_val = estimate_threshold(
        model, centroids, val_dir, args.data_dir, device
    )

    # Write threshold back to checkpoint
    if os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt["threshold"] = threshold
        torch.save(ckpt, args.checkpoint)
        logger.info(f"Threshold {threshold:.4f} written back to checkpoint")

    # ── End-of-training summary ───────────────────────────────────────────
    last_stats = stats_list[-1] if stats_list else {}
    backup_display = backup_path if os.path.isfile(args.checkpoint) else "N/A"

    print("\n" + "=" * 60)
    print(f"{'Scenario':<22}: {scenario} ({'fine-tune from checkpoint' if scenario == 'B' else 'train from scratch'})")
    print(f"{'Loss function':<22}: {args.loss}")
    print(f"{'Classes':<22}: {num_classes}")
    print(f"{'Total params':<22}: {total_params:,}")
    print(f"{'Trainable':<22}: {trainable_params:,}")
    print(f"{'Best Recall@1':<22}: {best_r1:.3f}")
    print(f"{'Best epoch':<22}: {best_epoch}")
    print(f"{'Previous best':<22}: {previous_best_r1:.3f}")
    delta = best_r1 - previous_best_r1
    print(f"{'Delta':<22}: {delta:+.3f}")
    print(f"{'Final Intra dist':<22}: {last_stats.get('mean_intra_class_distance', 'N/A')}")
    print(f"{'Final Inter dist':<22}: {last_stats.get('mean_inter_class_distance', 'N/A')}")
    print(f"{'Seed':<22}: {args.seed}")
    print(f"{'Backup saved':<22}: {backup_display}")
    print("=" * 60 + "\n")

    # ── Smoke test ────────────────────────────────────────────────────────
    if not args.no_smoke_test:
        run_smoke_test(args.checkpoint, args.gallery_path, args.data_dir, node_id_set)
    else:
        logger.info("Smoke test skipped (--no_smoke_test)")


if __name__ == "__main__":
    main()

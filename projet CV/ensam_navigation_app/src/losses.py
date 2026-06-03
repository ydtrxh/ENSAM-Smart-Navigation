"""
losses.py
=========
Loss functions for metric learning:
  - BatchHardTripletLoss  (baseline — current setup)
  - ArcFaceHead           (additive angular margin head, training-only)
  - CombinedLoss          (triplet + ArcFace, weight controlled by arcface_weight)

The ArcFaceHead is a classification head attached during training only.
After training it is discarded — the 128-dim embedding head is what gets saved and used.

Usage in retrain.py:
    --loss triplet    → BatchHardTripletLoss only
    --loss arcface    → ArcFaceHead only
    --loss combined   → triplet_loss + arcface_weight * arcface_loss
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet mining (Hermans et al., 2017).
    Uses cosine distance (1 − cos_sim) because embeddings are L2-normalized.

    Returns:
        mean_loss       (scalar)  — mean triplet loss over the batch
        active_pct      (float)   — fraction of triplets with loss > 0
        hard_neg_pct    (float)   — fraction of negatives with d(a,n) < margin
                                    (distinct from active_pct)
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        B = embeddings.size(0)

        # Cosine similarity matrix (valid because embeddings are L2-normalized)
        sim_matrix  = torch.matmul(embeddings, embeddings.t())
        dist_matrix = 1.0 - sim_matrix  # cosine distance

        labels_eq   = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B) bool
        labels_neq  = ~labels_eq

        diag_mask    = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        positive_mask = labels_eq & ~diag_mask

        # ── Hardest positives (max dist within class) ─────────────────────
        pos_dist = dist_matrix.clone()
        pos_dist[~positive_mask] = -1e9
        hardest_pos_dist, _ = pos_dist.max(dim=1)   # (B,)

        # ── Hardest negatives (min dist across classes) ────────────────────
        neg_dist = dist_matrix.clone()
        neg_dist[~labels_neq] = 1e9
        hardest_neg_dist, _ = neg_dist.min(dim=1)   # (B,)

        # ── Triplet loss ───────────────────────────────────────────────────
        triplet_loss = torch.clamp(
            hardest_pos_dist - hardest_neg_dist + self.margin, min=0.0
        )

        active_pct  = (triplet_loss > 0).float().mean().item()
        # Hard negative: chosen negative is within margin distance of anchor
        hard_neg_pct = (hardest_neg_dist < self.margin).float().mean().item()

        mean_loss = triplet_loss.mean()
        return mean_loss, active_pct, hard_neg_pct


class ArcFaceHead(nn.Module):
    """
    ArcFace — Additive Angular Margin Loss (Deng et al., 2019).

    This head is attached to the 128-dim embedding output during training only.
    After training, it is discarded — only the embedding backbone is saved.

    Args:
        in_features:  dimensionality of the input embedding (128)
        num_classes:  number of building classes
        margin:       additive angular margin m (default: 0.5 radians)
        scale:        feature scale s (default: 64.0)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.num_classes  = num_classes
        self.margin       = margin
        self.scale        = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)   # threshold: cos(π - m)
        self.mm    = math.sin(math.pi - margin) * margin  # sin(π-m)*m

        self.ce_loss = nn.CrossEntropyLoss()

        logger.debug(
            f"ArcFaceHead: in={in_features} classes={num_classes} "
            f"margin={margin:.3f} scale={scale:.1f}"
        )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized (B, 128)
            labels:     long tensor (B,)
        Returns:
            ArcFace cross-entropy loss (scalar)
        """
        # Weight vectors must also be L2-normalized for the cosine interpretation
        w_norm  = F.normalize(self.weight, p=2, dim=1)   # (C, 128)
        cosine  = F.linear(embeddings, w_norm)            # (B, C)

        sine    = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-6))
        phi     = cosine * self.cos_m - sine * self.sin_m   # cos(θ + m)

        # If θ + m > π, use linear approximation to keep monotonicity
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot for target classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Replace target logit with phi, keep others as cosine
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        return self.ce_loss(logits, labels.long())


class CombinedLoss(nn.Module):
    """
    Combined loss: triplet_loss + arcface_weight * arcface_loss.

    Args:
        num_classes:    number of building classes (for ArcFaceHead)
        embedding_dim:  128 (must match model output)
        margin:         triplet margin (default: 0.3)
        arcface_margin: ArcFace angular margin (default: 0.5)
        arcface_scale:  ArcFace scale (default: 64.0)
        arcface_weight: weight w for arcface term (default: 0.5)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        margin: float = 0.3,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        arcface_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.arcface_weight = arcface_weight
        self.triplet  = BatchHardTripletLoss(margin=margin)
        self.arcface  = ArcFaceHead(
            in_features=embedding_dim,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale,
        )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Returns:
            total_loss    (scalar)
            active_pct    (float)
            hard_neg_pct  (float)
        """
        triplet_loss, active_pct, hard_neg_pct = self.triplet(embeddings, labels)
        arc_loss  = self.arcface(embeddings, labels)
        total     = triplet_loss + self.arcface_weight * arc_loss
        return total, active_pct, hard_neg_pct


def build_loss(
    loss_name: str,
    num_classes: int,
    margin: float,
    arcface_margin: float,
    arcface_scale: float,
    arcface_weight: float,
    device: torch.device,
) -> nn.Module:
    """
    Factory function.  Returns the loss module and logs the active configuration.

    Args:
        loss_name: "triplet" | "arcface" | "combined"
    """
    if loss_name == "triplet":
        logger.info(f"Loss function  : triplet (batch-hard, margin={margin:.2f})")
        return BatchHardTripletLoss(margin=margin).to(device)

    elif loss_name == "arcface":
        logger.info(
            f"Loss function  : arcface (m={arcface_margin:.2f}, s={arcface_scale:.1f})"
        )
        # Wrap ArcFaceHead in a module that matches the (loss, act, hn) return signature
        class ArcFaceOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = ArcFaceHead(128, num_classes, arcface_margin, arcface_scale)
            def forward(self, emb, labels):
                loss = self.head(emb, labels)
                return loss, 0.0, 0.0

        return ArcFaceOnly().to(device)

    elif loss_name == "combined":
        logger.info(
            f"Loss function  : combined (triplet + arcface, "
            f"w={arcface_weight:.2f}, m={arcface_margin:.2f}, s={arcface_scale:.1f})"
        )
        return CombinedLoss(
            num_classes=num_classes,
            embedding_dim=128,
            margin=margin,
            arcface_margin=arcface_margin,
            arcface_scale=arcface_scale,
            arcface_weight=arcface_weight,
        ).to(device)

    else:
        raise ValueError(f"Unknown loss: {loss_name!r}. Choose from: triplet, arcface, combined")

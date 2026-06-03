import torch
import torch.nn as nn

class BatchHardTripletLoss(nn.Module):
    """
    Implements batch-hard triplet mining following Hermans et al. (2017).
    Uses cosine distance since embeddings are assumed to be L2-normalized.
    """
    def __init__(self, margin=0.3):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (B, D), L2-normalized.
            labels: Tensor of shape (B,)
            
        Returns:
            mean_loss: Mean triplet loss.
            active_fraction: Fraction of triplets with loss > 0.
        """
        # Cosine similarity matrix. Since embeddings are L2 normalized, 
        # dot product is equivalent to cosine similarity.
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Cosine distance: 1 - cosine_similarity
        dist_matrix = 1.0 - sim_matrix
        
        # Create boolean masks for positives and negatives
        labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))
        labels_not_equal = ~labels_equal
        
        # Remove the diagonal from the positive mask (don't compare item with itself)
        batch_size = embeddings.size(0)
        mask_diag = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        positive_mask = labels_equal & ~mask_diag
        
        # 1. Hardest Positives (maximum distance among positives)
        # We set distances of non-positives to a very small number so they aren't picked
        pos_dist_matrix = dist_matrix.clone()
        pos_dist_matrix[~positive_mask] = -1e12
        hardest_positive_dist, _ = pos_dist_matrix.max(dim=1)
        
        # 2. Hardest Negatives (minimum distance among negatives)
        # We set distances of non-negatives to a very large number so they aren't picked
        max_dist = dist_matrix.max().item()
        neg_dist_matrix = dist_matrix.clone()
        neg_dist_matrix[~labels_not_equal] = max_dist + 1e12
        hardest_negative_dist, _ = neg_dist_matrix.min(dim=1)
        
        # 3. Triplet Loss Computation
        # Loss = max(0, d_ap - d_an + margin)
        triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0)
        
        # Calculate active triplet fraction (where loss > 0)
        active_triplets = (triplet_loss > 0)
        active_count = active_triplets.sum().item()
        active_fraction = active_count / batch_size if batch_size > 0 else 0.0
        
        # Mean loss over the batch
        mean_loss = triplet_loss.mean()
        
        return mean_loss, active_fraction

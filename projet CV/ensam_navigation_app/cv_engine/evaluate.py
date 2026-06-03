import os
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.manifold import TSNE

from .dataset import ReferenceGalleryDataset
from .model import MetricLearningModel

def build_gallery(model, gallery_loader, device):
    """
    Builds the reference gallery centroids using double normalization.
    """
    model.eval()
    class_embeds = {}
    
    with torch.no_grad():
        for images, class_names in gallery_loader:
            images = images.to(device)
            # 1. Compute embedding (which is L2-normalized inside the model's forward pass)
            embeds = model(images).cpu()
            
            for i, cname in enumerate(class_names):
                if cname not in class_embeds:
                    class_embeds[cname] = []
                class_embeds[cname].append(embeds[i])
                
    centroids = {}
    for cname, emb_list in class_embeds.items():
        # Double normalization explanation:
        # Averaging non-normalized embeddings then normalizing produces a different (worse) 
        # centroid than averaging already-normalized embeddings. We want the mean direction
        # on the unit hypersphere. The model output is already L2-normalized.
        # We average these points, which pulls the vector inside the hypersphere, 
        # then L2-normalize the resulting class centroid before storage to project it back.
        stacked = torch.stack(emb_list)
        avg = stacked.mean(dim=0)
        centroid = torch.nn.functional.normalize(avg.unsqueeze(0), p=2, dim=1).squeeze(0)
        centroids[cname] = centroid
        
    return centroids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to save outputs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data Split
    from sklearn.model_selection import train_test_split
    image_paths = []
    class_labels = []
    classes = sorted(os.listdir(args.data_dir))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(args.data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                class_labels.append(class_name)

    train_paths, val_paths, train_lbls, val_lbls = train_test_split(
        image_paths, class_labels, test_size=0.2, stratify=class_labels, random_state=42
    )

    # 2. Load Model
    model = MetricLearningModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # We need a custom dataset class since ReferenceGalleryDataset expects a directory
    from .dataset import get_val_transforms
    from torch.utils.data import Dataset
    from PIL import Image
    class PathDataset(Dataset):
        def __init__(self, paths, labels, transforms):
            self.paths = paths
            self.labels = labels
            self.transforms = transforms
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            if self.transforms: img = self.transforms(img)
            return img, self.labels[idx]

    # 3. Build Gallery
    gallery_dataset = PathDataset(train_paths, train_lbls, get_val_transforms())
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)
    print("Building reference gallery...")
    centroids = build_gallery(model, gallery_loader, device)
    
    # Save gallery
    gallery_path = os.path.join(args.output_dir, 'gallery.pkl')
    with open(gallery_path, 'wb') as f:
        import pickle
        pickle.dump(centroids, f)
    print(f"Saved gallery to {gallery_path}")
    
    # 4. Evaluate Test Set
    test_dataset = PathDataset(val_paths, val_lbls, get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    all_embeds = []
    all_true_labels = []
    
    with torch.no_grad():
        for images, class_names in test_loader:
            images = images.to(device)
            embeds = model(images).cpu()
            all_embeds.append(embeds)
            all_true_labels.extend(class_names)
            
    all_embeds = torch.cat(all_embeds, dim=0)
    
    # Compute similarities to all centroids
    gallery_classes = sorted(centroids.keys())
    centroid_tensor = torch.stack([centroids[c] for c in gallery_classes])
    
    # Cosine similarities
    sim_matrix = torch.matmul(all_embeds, centroid_tensor.t())
    
    # Top-K matches
    k_max = min(5, len(gallery_classes))
    topk_sims, topk_indices = sim_matrix.topk(k=k_max, dim=1)
    
    topk_preds = []
    for i in range(len(all_embeds)):
        preds = [gallery_classes[idx.item()] for idx in topk_indices[i]]
        topk_preds.append(preds)
        
    # Metrics
    y_true = all_true_labels
    y_pred = [p[0] for p in topk_preds]
    max_sims = topk_sims[:, 0].numpy()
    
    correct_at_1 = sum(1 for true, preds in zip(y_true, topk_preds) if true == preds[0])
    correct_at_3 = sum(1 for true, preds in zip(y_true, topk_preds) if true in preds[:3])
    correct_at_5 = sum(1 for true, preds in zip(y_true, topk_preds) if true in preds[:5])
    
    N = len(y_true)
    recall_1 = correct_at_1 / N
    recall_3 = correct_at_3 / N
    recall_5 = correct_at_5 / N
    
    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    print(f"Top-1 Accuracy / Recall@1: {recall_1:.4f}")
    print(f"Recall@3: {recall_3:.4f}")
    print(f"Recall@5: {recall_5:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, xticks_rotation='vertical')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # t-SNE Plot
    print("\nGenerating t-SNE plot...")
    # Perplexity must be less than number of samples
    perplexity = min(15, len(all_embeds) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeds_2d = tsne.fit_transform(all_embeds.numpy())
    
    plt.figure(figsize=(10, 8))
    unique_classes = list(set(y_true))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
    
    for cls, color in zip(unique_classes, colors):
        idx = [i for i, lbl in enumerate(y_true) if lbl == cls]
        plt.scatter(embeds_2d[idx, 0], embeds_2d[idx, 1], label=cls, c=[color], alpha=0.7)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.title('t-SNE of Test Embeddings')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'tsne_plot.png'))
    plt.close()
    
    # 4. Auto-estimate unknown threshold
    print("\nAuto-estimating optimal cosine similarity threshold...")
    is_correct = np.array([1 if true == pred else 0 for true, pred in zip(y_true, y_pred)])
    
    thresholds = np.linspace(0.0, 1.0, 100)
    best_f1 = -1.0
    best_thresh = 0.6
    
    for t in thresholds:
        # Predict 1 (correct class match) if sim >= t, else 0 (unknown)
        preds_t = (max_sims >= t).astype(int)
        f1 = f1_score(is_correct, preds_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"Optimal threshold found: {best_thresh:.3f} (F1 Score: {best_f1:.3f})")
    
    # Overwrite threshold in checkpoint
    checkpoint['threshold'] = best_thresh
    torch.save(checkpoint, args.checkpoint)
    print(f"Updated checkpoint {args.checkpoint} with new threshold.")

if __name__ == "__main__":
    main()

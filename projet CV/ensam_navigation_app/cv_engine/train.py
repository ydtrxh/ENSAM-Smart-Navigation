import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .dataset import CampusDataset, get_train_transforms, get_val_transforms, PKSampler
from .model import MetricLearningModel
from .loss import BatchHardTripletLoss

def evaluate(model, train_loader, val_loader, device, criterion):
    """
    Evaluates validation set using nearest class centroid on training set embeddings.
    Returns: val_loss, val_active_frac, recall_1, recall_3, recall_5
    """
    model.eval()
    
    # 1. Build training centroids (mini-gallery)
    class_embeds = {}
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            embeds = model(images).cpu()
            for i in range(len(labels)):
                lbl = labels[i].item()
                if lbl not in class_embeds:
                    class_embeds[lbl] = []
                class_embeds[lbl].append(embeds[i])
                
    # Double normalization: average L2-normalized embeddings, then L2-normalize the centroid
    centroids = {}
    for lbl, emb_list in class_embeds.items():
        stacked = torch.stack(emb_list) # (N, 128)
        avg = stacked.mean(dim=0)
        avg = torch.nn.functional.normalize(avg.unsqueeze(0), p=2, dim=1).squeeze(0)
        centroids[lbl] = avg
        
    sorted_classes = sorted(centroids.keys())
    centroid_tensor = torch.stack([centroids[lbl] for lbl in sorted_classes]).to(device)
    centroid_labels = torch.tensor(sorted_classes).to(device)
    
    # 2. Evaluate validation set
    all_val_embeds = []
    all_val_labels = []
    val_loss_sum = 0.0
    active_fraction_sum = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            embeds = model(images)
            
            loss, active_frac = criterion(embeds, labels)
            val_loss_sum += loss.item() * len(images)
            active_fraction_sum += active_frac * len(images)
            
            all_val_embeds.append(embeds)
            all_val_labels.append(labels)
            
    all_val_embeds = torch.cat(all_val_embeds, dim=0)
    all_val_labels = torch.cat(all_val_labels, dim=0)
    
    # Cosine similarities to centroids
    sim_matrix = torch.matmul(all_val_embeds, centroid_tensor.t())
    
    # Compute Recall@K
    # Use min(5, num_classes) for k to avoid errors if classes < 5
    k_max = min(5, len(sorted_classes))
    _, topk_indices = sim_matrix.topk(k=k_max, dim=1)
    topk_preds = centroid_labels[topk_indices]
    
    correct_at_1 = (topk_preds[:, 0] == all_val_labels).sum().item()
    
    correct_at_3 = 0
    if k_max >= 3:
        correct_at_3 = (topk_preds[:, :3] == all_val_labels.unsqueeze(1)).any(dim=1).sum().item()
    else:
        correct_at_3 = correct_at_1
        
    correct_at_5 = 0
    if k_max >= 5:
        correct_at_5 = (topk_preds[:, :5] == all_val_labels.unsqueeze(1)).any(dim=1).sum().item()
    else:
        correct_at_5 = correct_at_3
    
    V = len(all_val_labels)
    recall_1 = correct_at_1 / V
    recall_3 = correct_at_3 / V
    recall_5 = correct_at_5 / V
    
    mean_loss = val_loss_sum / V
    mean_active = active_fraction_sum / V
    
    return mean_loss, mean_active, recall_1, recall_3, recall_5

def plot_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Triplet Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    # Plot Recalls
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_recall_1'], label='Recall@1')
    plt.plot(history['val_recall_3'], label='Recall@3')
    plt.plot(history['val_recall_5'], label='Recall@5')
    plt.title('Validation Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'recall_curve.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to save plots")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--p", type=int, default=5, help="Number of classes per batch (P)")
    parser.add_argument("--k", type=int, default=4, help="Number of images per class per batch (K)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    image_paths = []
    class_labels = []
    class_to_idx = {}
    
    classes = sorted(os.listdir(args.data_dir))
    for idx, class_name in enumerate(classes):
        class_to_idx[class_name] = idx
        class_dir = os.path.join(args.data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                class_labels.append(idx)
                
    if len(image_paths) == 0:
        print(f"No images found in {args.data_dir}")
        return

    # 80/20 train/validation split stratified by class
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, class_labels, test_size=0.2, stratify=class_labels, random_state=42
    )
    
    train_dataset = CampusDataset(train_paths, train_labels, transforms=get_train_transforms())
    val_dataset = CampusDataset(val_paths, val_labels, transforms=get_val_transforms())
    
    # We need an unaugmented train loader for building validation centroids
    train_dataset_eval = CampusDataset(train_paths, train_labels, transforms=get_val_transforms())
    train_eval_loader = DataLoader(train_dataset_eval, batch_size=32, shuffle=False, num_workers=0)

    # Use PKSampler for training
    try:
        train_sampler = PKSampler(train_labels, p=args.p, k=args.k)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    except ValueError as e:
        print(f"Sampler error: {e}. Trying smaller P.")
        args.p = min(args.p, len(set(train_labels)))
        train_sampler = PKSampler(train_labels, p=args.p, k=args.k)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 2. Setup Model, Loss, Optimizer
    model = MetricLearningModel().to(device)
    criterion = BatchHardTripletLoss(margin=0.3)
    
    # Differential learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Training Loop
    history = {'train_loss': [], 'val_loss': [], 'val_recall_1': [], 'val_recall_3': [], 'val_recall_5': []}
    best_recall = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        active_frac_sum = 0.0
        steps = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            embeds = model(images)
            loss, active_frac = criterion(embeds, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            active_frac_sum += active_frac
            steps += 1
            
        scheduler.step()
        
        avg_train_loss = train_loss_sum / steps if steps > 0 else 0
        avg_active_frac = active_frac_sum / steps if steps > 0 else 0
        
        # Validation
        val_loss, val_active, r1, r3, r5 = evaluate(model, train_eval_loader, val_loader, device, criterion)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_recall_1'].append(r1)
        history['val_recall_3'].append(r3)
        history['val_recall_5'].append(r5)
        
        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f} (Active: {avg_active_frac:.2f}) | "
              f"Val Loss: {val_loss:.4f} | R@1: {r1:.4f} R@3: {r3:.4f} R@5: {r5:.4f}")
              
        # Early Stopping & Checkpointing
        if r1 > best_recall:
            best_recall = r1
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall_1': best_recall,
                'threshold': 0.6  # Default initial threshold
            }, checkpoint_path)
            print(f" -> Best model saved at epoch {epoch} with R@1: {r1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    # 4. End of Training Summary & Plots
    plot_curves(history, args.output_dir)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*40)
    print("Training Summary")
    print("="*40)
    print(f"Total Parameters: {frozen_params + trainable_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Best Val Recall@1: {best_recall:.4f}")
    print(f"Achieved at Epoch: {best_epoch}")
    print(f"Loss curves saved to: {args.output_dir}")
    print("="*40)

if __name__ == "__main__":
    main()

import os
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T

# ImageNet means and stds
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms():
    """
    Returns the augmentation pipeline for training.
    """
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomPerspective(distortion_scale=0.3),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Note: RandomErasing must come AFTER ToTensor because it operates 
        # on PyTorch tensors, not on PIL Images.
        T.RandomErasing(p=0.2)
    ])

def get_val_transforms():
    """
    Returns the augmentation pipeline for validation and inference.
    """
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

class CampusDataset(Dataset):
    """
    Dataset for training and validation.
    Expects a list of (image_path, class_id) tuples.
    """
    def __init__(self, image_paths, class_ids, transforms=None):
        self.image_paths = image_paths
        self.class_ids = class_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.class_ids[idx]
        
        # We use PIL for image loading. opencv-python is available if needed.
        image = Image.open(path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, label

class ReferenceGalleryDataset(Dataset):
    """
    Dataset used exclusively for inference (building the gallery).
    Loads images from folder structure and returns (image_tensor, class_name).
    """
    def __init__(self, gallery_dir):
        self.gallery_dir = gallery_dir
        self.image_paths = []
        self.class_names = []
        self.transforms = get_val_transforms()
        
        for class_name in sorted(os.listdir(gallery_dir)):
            class_dir = os.path.join(gallery_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.class_names.append(class_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        class_name = self.class_names[idx]
        image = Image.open(path).convert('RGB')
        image = self.transforms(image)
        return image, class_name

class PKSampler(Sampler):
    """
    Custom sampler that ensures each batch contains exactly P classes 
    and K images per class. This guarantees valid anchor-positive pairs 
    for every sample in the batch, which is necessary for triplet loss.
    """
    def __init__(self, class_ids, p, k):
        self.class_ids = class_ids
        self.p = p
        self.k = k
        self.batch_size = p * k
        
        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx, cid in enumerate(self.class_ids):
            self.class_to_indices[cid].append(idx)
            
        self.classes = list(self.class_to_indices.keys())
        
        # Ensure we have enough classes
        if len(self.classes) < self.p:
            raise ValueError(f"Need at least {self.p} classes to form a batch, but got {len(self.classes)}")

    def __iter__(self):
        # Create a copy of class indices to sample from
        class_to_indices = {c: list(idxs) for c, idxs in self.class_to_indices.items()}
        for idxs in class_to_indices.values():
            random.shuffle(idxs)
            
        available_classes = list(self.classes)
        
        # Generate batches until we exhaust the dataset
        while len(available_classes) >= self.p:
            # Sample P classes without replacement
            sampled_classes = random.sample(available_classes, self.p)
            batch = []
            
            for c in sampled_classes:
                # Get K samples from this class
                class_idxs = class_to_indices[c]
                
                # If a class doesn't have enough samples left, we can oversample 
                # (sample with replacement) from the original list for this batch
                if len(class_idxs) < self.k:
                    # Fill with remaining and then oversample from all class indices
                    selected = class_idxs + random.choices(self.class_to_indices[c], k=self.k - len(class_idxs))
                    class_to_indices[c] = [] # Depleted
                else:
                    selected = class_idxs[:self.k]
                    class_to_indices[c] = class_idxs[self.k:]
                    
                batch.extend(selected)
                
                # If depleted, remove from available classes
                if len(class_to_indices[c]) == 0 and c in available_classes:
                    available_classes.remove(c)
                    
            yield batch

    def __len__(self):
        # Rough estimate of number of batches
        return len(self.class_ids) // self.batch_size

"""
Inference API for Metric-Learning Computer Vision Module.

Class Expansion Policy (Zero-Retraining):
------------------------------------------
This model is designed to support adding new building classes dynamically without ANY retraining.
To add a new building class:
1. Create a new folder with the class name under your gallery directory (e.g., `data/gallery/new_building/`).
2. Add at least one (preferably 3-5) reference images of the new building's entrance to this folder.
3. Rerun `evaluate.py` to rebuild the gallery `gallery.pkl` from the updated directory.
4. The embedding model remains frozen; only the gallery index is updated. Subsequent inference 
   calls will automatically recognize the new building without any changes to the network weights.
"""

import os
import argparse
import pickle
import torch
from PIL import Image

from .dataset import get_val_transforms
from .model import MetricLearningModel


def _load_image(source) -> Image.Image:
    """
    Flexible image loader that accepts:
      - str / os.PathLike  : opens from disk
      - PIL.Image.Image    : returned as-is (converted to RGB)
      - file-like object   : Streamlit UploadedFile, BytesIO, etc.
    """
    if isinstance(source, Image.Image):
        return source.convert('RGB')
    if isinstance(source, (str, os.PathLike)):
        return Image.open(source).convert('RGB')
    # Treat as a file-like object (UploadedFile, BytesIO, …)
    return Image.open(source).convert('RGB')


_MODEL_CACHE = None
_GALLERY_CACHE = None

def _get_model_and_gallery(gallery_path, checkpoint_path, device):
    global _MODEL_CACHE, _GALLERY_CACHE

    checkpoint_key = (os.path.abspath(checkpoint_path), os.path.getmtime(checkpoint_path))
    gallery_key = (os.path.abspath(gallery_path), os.path.getmtime(gallery_path))

    if _MODEL_CACHE is None or _MODEL_CACHE[0] != checkpoint_key:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        threshold = checkpoint.get('threshold', 0.6)
        model = MetricLearningModel().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _MODEL_CACHE = (checkpoint_key, model, threshold)
        
    if _GALLERY_CACHE is None or _GALLERY_CACHE[0] != gallery_key:
        with open(gallery_path, 'rb') as f:
            centroids = pickle.load(f)
        gallery_classes = sorted(centroids.keys())
        centroid_tensor = torch.stack([centroids[c] for c in gallery_classes]).to(device)
        _GALLERY_CACHE = (gallery_key, gallery_classes, centroid_tensor)
        
    return _MODEL_CACHE[1], _MODEL_CACHE[2], _GALLERY_CACHE[1], _GALLERY_CACHE[2]

def predict(image_source, gallery_path, checkpoint_path, device=None):
    """
    Predicts the class of a single image using the prebuilt gallery.
    """
    if device is None:
        device = torch.device("cpu") # Force CPU to avoid silent CUDA crashes on Windows
        
    # Load Cached Model and Gallery
    model, threshold, gallery_classes, centroid_tensor = _get_model_and_gallery(gallery_path, checkpoint_path, device)
    
    # Process Image — accepts a path, PIL.Image, or a file-like (e.g. Streamlit UploadedFile)
    transforms = get_val_transforms()
    image = _load_image(image_source)
    image_tensor = transforms(image).unsqueeze(0).to(device)
    
    # Extract Embedding
    with torch.no_grad():
        embedding = model(image_tensor)
        
    # Compare against gallery
    sim_matrix = torch.matmul(embedding, centroid_tensor.t())
    max_sim, max_idx = sim_matrix.max(dim=1)
    
    score = max_sim.item()
    pred_class = gallery_classes[max_idx.item()]
    
    # Threshold check
    if score < threshold:
        return "unknown", score
        
    return pred_class, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Campus Navigation")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--gallery_path", type=str, required=True, help="Path to gallery.pkl")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    args = parser.parse_args()
    
    pred_class, score = predict(args.image_path, args.gallery_path, args.checkpoint)
    print(f"Prediction: {pred_class} (Similarity Score: {score:.4f})")

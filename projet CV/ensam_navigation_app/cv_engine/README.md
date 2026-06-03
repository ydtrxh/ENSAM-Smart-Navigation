# Metric-Learning Computer Vision Module for ENSAM Navigation

This module implements a metric-learning computer vision system to identify campus buildings from photos of their doors or entrances.

## Architecture
The system uses a shared-weight EfficientNet-B0 encoder trained with Batch-Hard Triplet Loss to learn a 128-dimensional L2-normalized embedding space. During inference, it performs nearest-neighbor retrieval using cosine similarity against a prebuilt reference gallery.

There is no softmax classifier layer, making the system highly adaptable to new building classes without needing to retrain the neural network weights.

## Project Workflow

1. **Data Preparation**
   Organize your images into class-named subdirectories (e.g., `data/train/building_a/img1.jpg`).
   You need a very small dataset: approximately 3–10 images per class.

2. **Training (`train.py`)**
   Train the model using the metric learning approach.
   ```bash
   python train.py --data_dir ../data/train --output_dir outputs --checkpoint_dir checkpoints
   ```
   This will train the backbone, implement PK-sampling to mine hard triplets, and save the best checkpoint to `checkpoints/best_model.pth`.

3. **Evaluation and Gallery Building (`evaluate.py`)**
   Build the reference gallery and test the model's accuracy. The threshold for "unknown" buildings is automatically computed here.
   ```bash
   python evaluate.py --test_dir ../data/test --gallery_dir ../data/gallery --checkpoint checkpoints/best_model.pth --output_dir outputs
   ```
   Outputs will include a confusion matrix, t-SNE plot, and a serialized `gallery.pkl`.

4. **Inference (`inference.py`)**
   Use the prebuilt gallery to run inference on single images.
   ```bash
   python inference.py --image_path sample.jpg --gallery_path outputs/gallery.pkl --checkpoint checkpoints/best_model.pth
   ```

## Class Expansion Policy (Zero-Retraining)

This model supports adding new buildings to recognize **with zero retraining**.

If a new building (e.g., "Amphi 3") needs to be added:
1. Create a new folder `Amphi_3/` inside your gallery directory (`data/gallery/`).
2. Add a few reference photos (3-5 images) of the Amphi 3 entrance into that folder.
3. Rerun `evaluate.py` or a dedicated gallery building script to re-create the `gallery.pkl` file with the new centroids.
4. The system will automatically start recognizing "Amphi 3" during inference, utilizing the newly updated gallery. The neural network remains frozen.

## Expected Performance

With the provided highly-augmented pipeline and triplet loss strategy, despite the small dataset (60-210 images total across 21 classes), the model is capable of clustering visual features efficiently. 
You can expect a **Recall@1 of approximately 75–90%** depending on the visual distinctiveness of the building entrances and the quality of the reference images.

## Exporting for Production

The `model.py` module includes utility functions `export_onnx` and `export_torchscript` to convert the PyTorch model for mobile or edge deployment.

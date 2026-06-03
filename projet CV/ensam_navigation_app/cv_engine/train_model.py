"""
CNN Training Script for ENSAM Campus Location Recognition.

DATASET STRUCTURE REQUIRED:
dataset/
├── buvette/          ← 50-100 photos of "Buvette des Etudiants" door
├── dept_mecanique/   ← 50-100 photos of "Génie Mécanique" door
├── labo_civil/       ← 50-100 photos of "Génie Civil" lab door
└── salles_td_1/      ← 50-100 photos of "Salles TD I" door

Run: python train_model.py
Output: campus_cnn_model.h5 + model_metrics.json (for the Streamlit app)
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import numpy as np
import json
import os
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25
DATASET_DIR = "dataset"
MODEL_PATH = "campus_cnn_model.h5"
METRICS_PATH = "model_metrics.json"

# ==============================================================================
# 1. LOAD AND SPLIT DATASET (80% train / 20% validation)
# ==============================================================================
print("🔄 Chargement du dataset...")

if not os.path.exists(DATASET_DIR):
    print(f"❌ ERREUR: Le dossier '{DATASET_DIR}' est introuvable.")
    print("   Créez ce dossier et ajoutez-y des sous-dossiers avec des images.")
    exit(1)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ {num_classes} classes détectées : {class_names}")

# Optimize for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================================================================
# 2. DATA AUGMENTATION (combats overfitting with small datasets)
# ==============================================================================
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),   # Simulates different lighting conditions
    layers.RandomContrast(0.2),
], name="data_augmentation")

# ==============================================================================
# 3. CNN MODEL ARCHITECTURE (MobileNet-inspired but custom)
# ==============================================================================
print("🏗️ Construction du modèle CNN...")

model = models.Sequential([
    # Input normalization
    layers.Rescaling(1./255, input_shape=(150, 150, 3)),
    
    # Data augmentation (only active during training)
    data_augmentation,
    
    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Convolutional Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Classifier Head
    layers.GlobalAveragePooling2D(),  # Better than Flatten for generalization
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
], name="ENSAM_Campus_CNN")

# ==============================================================================
# 4. COMPILATION
# ==============================================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================================================================
# 5. TRAINING WITH EARLY STOPPING
# ==============================================================================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,                  # Stop if val_accuracy doesn't improve for 5 epochs
        restore_best_weights=True,   # Revert to the best weights
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,         # Only save when val_accuracy improves
        verbose=1
    )
]

print(f"\n🚀 Début de l'entraînement ({EPOCHS} époques max, early stopping activé)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==============================================================================
# 6. EVALUATION & METRICS EXPORT (for Streamlit dashboard)
# ==============================================================================
print("\n📊 Évaluation finale sur les données de validation...")

y_true, y_pred = [], []
for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    for label, pred in zip(labels.numpy(), predictions):
        y_true.append(label)
        y_pred.append(np.argmax(pred))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

final_accuracy = float(np.mean(y_true == y_pred))
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

print(f"\n{'='*60}")
print(f"🎯 ACCURACY FINALE : {final_accuracy * 100:.2f}%")
print(f"{'='*60}")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save metrics as JSON for the Streamlit app to read
metrics_data = {
    "accuracy": final_accuracy,
    "class_names": class_names,
    "report": report,
    "epochs_trained": len(history.history['accuracy'])
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4, ensure_ascii=False)
print(f"\n💾 Métriques sauvegardées dans '{METRICS_PATH}'")
print(f"💾 Modèle sauvegardé dans '{MODEL_PATH}'")

# ==============================================================================
# 7. PLOT TRAINING CURVES
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#1E3A8A')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#10B981')
ax1.set_title('Accuracy par Époque')
ax1.set_xlabel('Époque')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(history.history['loss'], label='Train Loss', color='#DC2626')
ax2.plot(history.history['val_loss'], label='Val Loss', color='#F59E0B')
ax2.set_title('Loss par Époque')
ax2.set_xlabel('Époque')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("📈 Courbes d'entraînement sauvegardées dans 'training_curves.png'")
plt.show()
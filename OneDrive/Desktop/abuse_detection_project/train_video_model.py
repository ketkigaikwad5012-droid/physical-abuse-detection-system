import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    LSTM, TimeDistributed, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────
FRAMES_DIR  = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\video_frames"
IMG_SIZE    = (64, 64)
FRAMES      = 8
BATCH_SIZE  = 8
EPOCHS      = 30

# ── Load from saved .npy files (instant!) ─────────────────────
def load_dataset(frames_dir):
    X, y = [], []

    classes = {
        'violent':     1,
        'non_violent': 0
    }

    for class_name, label in classes.items():
        class_path = os.path.join(frames_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Not found: {class_path}")
            continue

        files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        print(f"  {class_name}: {len(files)} files")

        for npy_file in files:
            frames = np.load(os.path.join(class_path, npy_file))
            X.append(frames)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)

# ── Load dataset ──────────────────────────────────────────────
print("=== Loading saved frames (instant) ===")
X, y = load_dataset(FRAMES_DIR)

print(f"\nDataset loaded!")
print(f"  X shape      : {X.shape}")
print(f"  Violence     : {np.sum(y == 1)}")
print(f"  Non-violence : {np.sum(y == 0)}")

# ── Train/val split ───────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ── Build MobileNetV2 + LSTM ──────────────────────────────────
print("\n=== Building model ===")

base_model = MobileNetV2(
    input_shape=(64, 64, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = Input(shape=(FRAMES, 64, 64, 3))
x       = TimeDistributed(base_model)(inputs)
x       = TimeDistributed(GlobalAveragePooling2D())(x)
x       = LSTM(64, return_sequences=False)(x)
x       = Dropout(0.5)(x)
x       = Dense(32, activation='relu')(x)
x       = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Class weights ─────────────────────────────────────────────
total    = len(y_train)
n_v      = np.sum(y_train == 1)
n_nv     = np.sum(y_train == 0)
class_weight = {
    1: total / (2 * n_v)  if n_v  > 0 else 1.0,
    0: total / (2 * n_nv) if n_nv > 0 else 1.0
}
print(f"Class weights: {class_weight}")

# ── Callbacks ─────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=6,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ── Train ─────────────────────────────────────────────────────
print("\n=== Training ===")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr]
)

# ── Save ──────────────────────────────────────────────────────
model.save("violence_video_model.h5")
print("\nModel saved as violence_video_model.h5")

# ── Plot ──────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],     label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.title("Accuracy"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],     label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title("Loss"); plt.legend()

plt.tight_layout()
plt.savefig("video_training_results.png")
plt.show()

best_val = max(history.history['val_accuracy'])
print(f"\nBest val accuracy: {best_val*100:.1f}%")
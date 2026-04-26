import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, LSTM, TimeDistributed, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── PATHS ─────────────────────────────────────────────────────
BASE    = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project"
OUTPUT  = os.path.join(BASE, "combined_video_dataset")
FRAMES  = os.path.join(BASE, "combined_video_frames")

SOURCES = {
    'violent': [
        os.path.join(BASE, "A-Dataset-for-Automatic-Violence-Detection-in-Videos-master", "violence-detection-dataset", "violent", "cam1"),
        os.path.join(BASE, "A-Dataset-for-Automatic-Violence-Detection-in-Videos-master", "violence-detection-dataset", "violent", "cam2"),
        os.path.join(BASE, "Real Life Violence Dataset", "Violence"),
        os.path.join(BASE, "real life violence situations", "Real Life Violence Dataset", "Violence"),
        os.path.join(BASE, "VioPeru-main", "train", "Fight"),
        os.path.join(BASE, "VioPeru-main", "val",   "Fight"),
    ],
    'non_violent': [
        os.path.join(BASE, "A-Dataset-for-Automatic-Violence-Detection-in-Videos-master", "violence-detection-dataset", "non-violent", "cam1"),
        os.path.join(BASE, "A-Dataset-for-Automatic-Violence-Detection-in-Videos-master", "violence-detection-dataset", "non-violent", "cam2"),
        os.path.join(BASE, "Real Life Violence Dataset", "NonViolence"),
        os.path.join(BASE, "real life violence situations", "Real Life Violence Dataset", "NonViolence"),
        os.path.join(BASE, "VioPeru-main", "train", "NonFight"),
        os.path.join(BASE, "VioPeru-main", "val",   "NonFight"),
    ]
}

IMG_SIZE   = (64, 64)
NUM_FRAMES = 8
BATCH_SIZE = 8
EPOCHS     = 40

# ── STEP 1: COMBINE ALL DATASETS ──────────────────────────────
print("=" * 55)
print("STEP 1: Combining all video datasets")
print("=" * 55)

os.makedirs(f"{OUTPUT}/violent",     exist_ok=True)
os.makedirs(f"{OUTPUT}/non_violent", exist_ok=True)

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')
count_v = count_nv = 0

for label, paths in SOURCES.items():
    for src_path in paths:
        if not os.path.exists(src_path):
            print(f"  Skipping (not found): {src_path}")
            continue
        files = [f for f in os.listdir(src_path)
                 if f.lower().endswith(VIDEO_EXTS)]
        for f in files:
            src  = os.path.join(src_path, f)
            tag  = src_path.replace(BASE, "").replace("\\", "_")[:30]
            dst  = os.path.join(OUTPUT, label, f"{tag}_{f}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            if label == 'violent':
                count_v += 1
            else:
                count_nv += 1

print(f"\n  violent/     → {len(os.listdir(f'{OUTPUT}/violent'))} videos")
print(f"  non_violent/ → {len(os.listdir(f'{OUTPUT}/non_violent'))} videos")

# ── STEP 2: EXTRACT AND SAVE FRAMES ──────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Extracting frames (runs once only)")
print("=" * 55)

os.makedirs(f"{FRAMES}/violent",     exist_ok=True)
os.makedirs(f"{FRAMES}/non_violent", exist_ok=True)

def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total == 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = (frame / 255.0).astype(np.float32)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.array(frames[:num_frames])

for label in ['violent', 'non_violent']:
    src_dir  = os.path.join(OUTPUT, label)
    save_dir = os.path.join(FRAMES, label)
    videos   = [f for f in os.listdir(src_dir)
                if f.lower().endswith(VIDEO_EXTS)]
    done = 0
    for vf in videos:
        npy_name = vf.rsplit('.', 1)[0] + '.npy'
        npy_path = os.path.join(save_dir, npy_name)
        if os.path.exists(npy_path):
            done += 1
            continue
        try:
            frames = extract_frames(os.path.join(src_dir, vf))
            if frames is not None:
                np.save(npy_path, frames)
                done += 1
        except Exception as e:
            print(f"  Skipping corrupt file: {vf} — {e}")
            continue
    print(f"  {label}: {done} frame files saved")

# ── STEP 3: DATA GENERATOR (replaces loading all into RAM) ────
print("\n" + "=" * 55)
print("STEP 3: Setting up data generator")
print("=" * 55)

import random

def get_all_files(frames_dir):
    all_files = []
    for label, cls_id in [('violent', 1), ('non_violent', 0)]:
        fdir  = os.path.join(frames_dir, label)
        files = [f for f in os.listdir(fdir) if f.endswith('.npy')]
        for f in files:
            all_files.append((os.path.join(fdir, f), cls_id))
    random.shuffle(all_files)
    return all_files

all_files = get_all_files(FRAMES)
split     = int(len(all_files) * 0.8)
train_files = all_files[:split]
val_files   = all_files[split:]

print(f"  Train: {len(train_files)} | Val: {len(val_files)}")

# Count classes for weights
n_v  = sum(1 for _, l in train_files if l == 1)
n_nv = sum(1 for _, l in train_files if l == 0)
total = len(train_files)
class_weight = {
    1: total / (2 * n_v)  if n_v  > 0 else 1.0,
    0: total / (2 * n_nv) if n_nv > 0 else 1.0
}
print(f"  Violence: {n_v} | Non-violence: {n_nv}")
print(f"  Class weights: {class_weight}")

class VideoGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, batch_size=8, shuffle=True):
        self.file_list  = file_list
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.file_list) // self.batch_size

    def __getitem__(self, idx):
        batch = self.file_list[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        X, y = [], []
        for path, label in batch:
            try:
                arr = np.load(path)
                if arr.shape == (NUM_FRAMES, 64, 64, 3):
                    X.append(arr)
                    y.append(label)
            except:
                continue
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.file_list)

train_gen = VideoGenerator(train_files, batch_size=BATCH_SIZE)
val_gen   = VideoGenerator(val_files,   batch_size=BATCH_SIZE, shuffle=False)

# ── STEP 4: BUILD MODEL ───────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Building MobileNetV2 + LSTM model")
print("=" * 55)

base = MobileNetV2(
    input_shape=(64, 64, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False

inputs  = Input(shape=(NUM_FRAMES, 64, 64, 3))
x       = TimeDistributed(base)(inputs)
x       = TimeDistributed(GlobalAveragePooling2D())(x)
x       = LSTM(128, return_sequences=False)(x)
x       = Dropout(0.5)(x)
x       = Dense(64, activation='relu')(x)
x       = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)
model   = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── STEP 5: CALLBACKS ─────────────────────────────────────────
def make_callbacks(phase):
    return [
        EarlyStopping(
            monitor='val_accuracy', patience=7,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.3,
            patience=3, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            f'best_video_model_phase{phase}.h5',
            monitor='val_accuracy',
            save_best_only=True, verbose=1
        )
    ]

# ── STEP 6: PHASE 1 TRAINING ──────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Phase 1 — Training head only")
print("=" * 55)

h1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    class_weight=class_weight,
    callbacks=make_callbacks(1)
)

# ── STEP 7: PHASE 2 FINE-TUNING ───────────────────────────────
print("\n" + "=" * 55)
print("STEP 7: Phase 2 — Fine-tuning top layers")
print("=" * 55)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

h2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=make_callbacks(2)
)

# ── STEP 8: SAVE ──────────────────────────────────────────────
model.save("violence_video_model_v2.h5")
print("\nSaved: violence_video_model_v2.h5")

# ── STEP 9: PLOT ──────────────────────────────────────────────
acc  = h1.history['accuracy']     + h2.history['accuracy']
val  = h1.history['val_accuracy'] + h2.history['val_accuracy']
loss = h1.history['loss']         + h2.history['loss']
vloss= h1.history['val_loss']     + h2.history['val_loss']

plt.figure(figsize=(14, 5))
plt.subplot(1,2,1)
plt.plot(acc,  label='Train accuracy')
plt.plot(val,  label='Val accuracy')
plt.axvline(x=len(h1.history['accuracy'])-1,
            color='gray', linestyle='--', label='Fine-tune start')
plt.title("Accuracy"); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(loss,  label='Train loss')
plt.plot(vloss, label='Val loss')
plt.axvline(x=len(h1.history['loss'])-1,
            color='gray', linestyle='--', label='Fine-tune start')
plt.title("Loss"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig("video_training_v2_results.png")
plt.show()

print(f"\nBest val accuracy: {max(val)*100:.1f}%")
print("DONE!")
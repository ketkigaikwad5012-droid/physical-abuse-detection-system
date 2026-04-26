import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
DATASET    = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\dataset"

# ── Data loading ─────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    DATASET, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='training', seed=42
)
val_data = datagen.flow_from_directory(
    DATASET, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation', seed=42
)

print("Classes:", train_data.class_indices)

# ── Fix class imbalance ──────────────────────────────────────
total = 12013 + 8036
class_weight = {
    0: total / (2 * 8036),
    1: total / (2 * 12013)
}
print("Class weights:", class_weight)

# ── MobileNetV2 base ─────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# ── Custom head (stronger regularization) ────────────────────
inputs  = Input(shape=(224, 224, 3))
x       = base_model(inputs, training=False)
x       = GlobalAveragePooling2D()(x)
x       = Dense(128, activation='relu',
                kernel_regularizer=l2(0.001))(x)  # L2 regularization
x       = Dropout(0.6)(x)                          # stronger dropout
outputs = Dense(1, activation='sigmoid')(x)
model   = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ── Phase 1: Train head only ──────────────────────────────────
print("\n=== Phase 1: Training head only ===")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr]
)

# ── Phase 2: Fine-tune top layers ─────────────────────────────
print("\n=== Phase 2: Fine-tuning top layers ===")
base_model.trainable = True

# Freeze all except last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Much lower learning rate to avoid destroying pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Reset early stopping for phase 2
early_stop2 = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
reduce_lr2 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-8,
    verbose=1
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    class_weight=class_weight,
    callbacks=[early_stop2, reduce_lr2]
)

# ── Save ──────────────────────────────────────────────────────
model.save("violence_model_v2.h5")
print("\nModel saved as violence_model_v2.h5")

# ── Plot ──────────────────────────────────────────────────────
acc  = history1.history['accuracy']     + history2.history['accuracy']
val  = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss']         + history2.history['loss']
v_l  = history1.history['val_loss']     + history2.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(acc, label='Train accuracy')
plt.plot(val, label='Val accuracy')
plt.axvline(x=len(history1.history['accuracy'])-1,
            color='gray', linestyle='--', label='Fine-tune start')
plt.title("Accuracy"); plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='Train loss')
plt.plot(v_l,  label='Val loss')
plt.axvline(x=len(history1.history['loss'])-1,
            color='gray', linestyle='--', label='Fine-tune start')
plt.title("Loss"); plt.legend()

plt.tight_layout()
plt.savefig("training_results_v2.png")
plt.show()

best_val = max(val)
print(f"\nBest val accuracy: {best_val:.4f} ({best_val*100:.1f}%)")
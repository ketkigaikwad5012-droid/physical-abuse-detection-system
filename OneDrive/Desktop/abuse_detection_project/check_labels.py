import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Change this to your dataset path ─────────────────────────────────────────
DATASET_PATH = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\dataset"

# ─────────────────────────────────────────────────────────────────────────────
# CHECK FOLDER STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
print("Checking dataset folder structure...")
print(f"Path: {DATASET_PATH}\n")

if not os.path.exists(DATASET_PATH):
    print("❌ Dataset folder not found. Check DATASET_PATH above.")
    exit(1)

subfolders = [f for f in os.listdir(DATASET_PATH)
              if os.path.isdir(os.path.join(DATASET_PATH, f))]

print(f"Subfolders found: {subfolders}")

for folder in subfolders:
    folder_path = os.path.join(DATASET_PATH, folder)
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"  {folder}/  →  {len(images)} images")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK CLASS LABELS  — must match training setup
# ─────────────────────────────────────────────────────────────────────────────
print("\nChecking class label mapping (must match train_model.py)...")

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training",
)

print(f"\nClass labels: {train_data.class_indices}")
print(
    "\nExpected: {'non-violence': 0, 'violence': 1}"
    "\nIf labels are reversed the model will predict backwards — rename your folders to match."
)
print("\n✅ Check complete.")

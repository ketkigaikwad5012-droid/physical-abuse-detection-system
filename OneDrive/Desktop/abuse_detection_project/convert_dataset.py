import os
import shutil
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — change DATASET_ROOT to where your downloaded dataset folder is
# Example: r"C:\Users\KETKI\Downloads\violence-detection-p4qev-2"
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\violence-detection-2"
OUTPUT_PATH  = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\dataset"
 
# Class mapping from data.yaml
# 0 = non_violence, 1 = violence
CLASS_MAP = {
    "0": "non-violence",
    "1": "violence"
}
 
# ─────────────────────────────────────────────────────────────────────────────
# CREATE OUTPUT FOLDERS
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(f"{OUTPUT_PATH}/violence",     exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/non-violence", exist_ok=True)
print(f"Output folders created at: {OUTPUT_PATH}\n")
 
# ─────────────────────────────────────────────────────────────────────────────
# PROCESS EACH SPLIT  — train / valid / test
# For train: images are in train/images, labels in roboflow zip train folder
# For test:  images + labels both in test/images and test/labels
# ─────────────────────────────────────────────────────────────────────────────
 
# Map of split → (images folder, labels folder)
SPLITS = {
    "train": (
        os.path.join(DATASET_ROOT, "train", "images"),
        os.path.join(DATASET_ROOT, "train", "labels"),
    ),
    "valid": (
        os.path.join(DATASET_ROOT, "valid", "images"),
        os.path.join(DATASET_ROOT, "valid", "labels"),
    ),
    "test": (
        os.path.join(DATASET_ROOT, "test", "images"),
        os.path.join(DATASET_ROOT, "test", "labels"),
    ),
}
 
total_copied  = 0
total_skipped = 0
 
for split, (img_dir, lbl_dir) in SPLITS.items():
 
    if not os.path.exists(img_dir):
        print(f"⚠️  Skipping '{split}' — images folder not found: {img_dir}")
        continue
    if not os.path.exists(lbl_dir):
        print(f"⚠️  Skipping '{split}' — labels folder not found: {lbl_dir}")
        continue
 
    images = [f for f in os.listdir(img_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
 
    print(f"Processing {split}/ — {len(images)} images found")
 
    split_copied  = 0
    split_skipped = 0
 
    for img_file in images:
        # Find matching label file (.txt same stem)
        stem     = os.path.splitext(img_file)[0]
        lbl_file = os.path.join(lbl_dir, stem + ".txt")
 
        if not os.path.exists(lbl_file):
            split_skipped += 1
            continue
 
        # Read label file — take class from the first annotation line
        with open(lbl_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
 
        if not lines:
            split_skipped += 1
            continue
 
        class_id  = lines[0].split()[0]          # first token = class id
        class_name = CLASS_MAP.get(class_id)
 
        if class_name is None:
            print(f"  Unknown class id '{class_id}' in {lbl_file} — skipping")
            split_skipped += 1
            continue
 
        # Copy image to correct class folder
        src = os.path.join(img_dir, img_file)
        dst = os.path.join(OUTPUT_PATH, class_name, img_file)
 
        # Avoid overwriting if same filename exists from another split
        if os.path.exists(dst):
            base, ext = os.path.splitext(img_file)
            dst = os.path.join(OUTPUT_PATH, class_name, f"{split}_{base}{ext}")
 
        shutil.copy2(src, dst)
        split_copied += 1
 
    print(f"  ✅ Copied  : {split_copied}")
    print(f"  ⚠️  Skipped : {split_skipped}\n")
    total_copied  += split_copied
    total_skipped += split_skipped
 
# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
v_count   = len(os.listdir(f"{OUTPUT_PATH}/violence"))
nv_count  = len(os.listdir(f"{OUTPUT_PATH}/non-violence"))
 
print("═" * 50)
print("CONVERSION COMPLETE")
print(f"  violence/     → {v_count} images")
print(f"  non-violence/ → {nv_count} images")
print(f"  Total copied  : {total_copied}")
print(f"  Total skipped : {total_skipped}")
print(f"\nDataset ready at: {OUTPUT_PATH}")
print("═" * 50)
print("\nNext step → run: python check_labels.py")
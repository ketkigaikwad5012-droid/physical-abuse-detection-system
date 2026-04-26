import os
import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────
DATASET  = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\A-Dataset-for-Automatic-Violence-Detection-in-Videos-master\violence-detection-dataset"
SAVE_DIR = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\video_frames"
IMG_SIZE = (64, 64)
FRAMES   = 8

os.makedirs(f"{SAVE_DIR}/violent",     exist_ok=True)
os.makedirs(f"{SAVE_DIR}/non_violent", exist_ok=True)

def extract_and_save(video_path, save_path):
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames  = []

    if total == 0:
        cap.release()
        return False

    indices = np.linspace(0, total - 1, FRAMES, dtype=int)

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

    if len(frames) < FRAMES:
        while len(frames) < FRAMES:
            frames.append(frames[-1])

    # Save as .npy file — loads instantly next time
    np.save(save_path, np.array(frames))
    return True

# ── Process violent videos ────────────────────────────────────
print("=== Processing violent videos ===")
count = 0
for cam in ["cam1", "cam2"]:
    src = os.path.join(DATASET, "violent", cam)
    if not os.path.exists(src):
        print(f"Not found: {src}")
        continue
    for f in os.listdir(src):
        if f.lower().endswith('.mp4'):
            video_path = os.path.join(src, f)
            save_path  = os.path.join(
                SAVE_DIR, "violent", f"{cam}_{f.replace('.mp4', '.npy')}"
            )
            if extract_and_save(video_path, save_path):
                count += 1
                print(f"  Saved: {count} - {cam}/{f}")

# ── Process non-violent videos ────────────────────────────────
print("\n=== Processing non-violent videos ===")
count = 0
for cam in ["cam1", "cam2"]:
    src = os.path.join(DATASET, "non-violent", cam)
    if not os.path.exists(src):
        print(f"Not found: {src}")
        continue
    for f in os.listdir(src):
        if f.lower().endswith('.mp4'):
            video_path = os.path.join(src, f)
            save_path  = os.path.join(
                SAVE_DIR, "non_violent", f"{cam}_{f.replace('.mp4', '.npy')}"
            )
            if extract_and_save(video_path, save_path):
                count += 1
                print(f"  Saved: {count} - {cam}/{f}")

# ── Summary ───────────────────────────────────────────────────
v  = len(os.listdir(f"{SAVE_DIR}/violent"))
nv = len(os.listdir(f"{SAVE_DIR}/non_violent"))
print(f"\n=== DONE ===")
print(f"violent/     → {v} .npy files")
print(f"non_violent/ → {nv} .npy files")
print("\nNow run train_video_model.py — frames load instantly!")
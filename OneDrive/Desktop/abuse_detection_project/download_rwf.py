# save as download_rwf2.py
from huggingface_hub import snapshot_download
import sys

print("Starting RWF-2000 download...")
print("This is 12GB — will take 20-40 mins depending on speed")
print("Do NOT close this window!\n")

try:
    path = snapshot_download(
        repo_id="DanJoshua/RWF-2000",
        repo_type="dataset",
        local_dir=r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\RWF-2000",
        ignore_patterns=["*.md"],
        local_dir_use_symlinks=False
    )
    print(f"\nDownload complete! Saved to: {path}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
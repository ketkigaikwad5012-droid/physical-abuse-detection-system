import zipfile, os

zip_path  = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\violence-detection-2\roboflow.zip"
out_path  = r"C:\Users\KETKI\OneDrive\Desktop\abuse_detection_project\violence-detection-2"

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(out_path)

print("Extraction complete!")
from roboflow import Roboflow

print("Starting Roboflow download...")

rf = Roboflow(api_key="YOUR_API_KEY_HERE")

project = rf.workspace("securityviolence").project("violence-detection-p4qev")
dataset = project.version(2).download("yolov8")

print("Dataset downloaded successfully!")
print("Dataset location:", dataset.location)

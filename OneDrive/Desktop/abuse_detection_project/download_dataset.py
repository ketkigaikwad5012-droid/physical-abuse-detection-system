from roboflow import Roboflow

print("Starting Roboflow download...")

rf = Roboflow(api_key="L6H5SO6KfwCKv7i6OZLi")

project = rf.workspace("securityviolence").project("violence-detection-p4qev")
dataset = project.version(2).download("yolov8")

print("Dataset downloaded successfully!")
print("Dataset location:", dataset.location)
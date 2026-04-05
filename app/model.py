import os
from ultralytics import YOLO

# Get project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Model path
model_path = os.path.join(BASE_DIR, "best.pt")

# Load YOLO model once
model = YOLO(model_path)

print("Model loaded successfully from:", model_path)

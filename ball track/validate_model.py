import os

# Define paths
HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(HOME, "runs", "detect", "train", "weights", "best.pt")
DATASET_PATH = os.path.join(HOME, "datasets", "football-ball-detection-2", "data.yaml")

# Validate YOLOv8 model
os.system(f"""
yolo task=detect mode=val model={MODEL_PATH} data={DATASET_PATH} imgsz=1280
""")

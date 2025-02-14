import os
import torch
import torchvision
print(torch.__version__)  # Check PyTorch version
print(torchvision.__version__)  # Check Torchvision version
print(torch.cuda.is_available())  # Verify if CUDA is available


# Define paths
HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(HOME, "datasets", "football-ball-detection-2", "data.yaml")

# Train YOLOv8
os.system(f"""
yolo task=detect mode=train model=yolov8x.pt data={DATASET_PATH} batch=12 epochs=50 imgsz=1280 plots=True
""")

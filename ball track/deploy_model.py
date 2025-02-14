import os
from roboflow import Roboflow
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Initialize Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("roboflow-jvuqo").project("football-ball-detection-rejhg")
version = project.version(2)

# Define paths
HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(HOME, "runs", "detect", "train")

# Deploy model
version.deploy(model_type="yolov8", model_path=MODEL_PATH)

import os
import yaml  # Requires PyYAML, install with `pip install pyyaml`
from roboflow import Roboflow
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Set paths
HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(HOME, "datasets")

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.chdir(DATASET_DIR)

# Download dataset
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("roboflow-jvuqo").project("football-ball-detection-rejhg")
version = project.version(2)
dataset = version.download("yolov8")

# Path to data.yaml
data_yaml_path = os.path.join(dataset.location, "data.yaml")

# Update paths in data.yaml
with open(data_yaml_path, 'r') as file:
    data = yaml.safe_load(file)

# Modify the paths in the YAML
data['train'] = '../train/images'
data['val'] = '../valid/images'

# Save the updated YAML
with open(data_yaml_path, 'w') as file:
    yaml.safe_dump(data, file)

print(f"Updated data.yaml: {data_yaml_path}")

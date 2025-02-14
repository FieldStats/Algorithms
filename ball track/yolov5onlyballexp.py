import cv2
import torch
import warnings
from tqdm import tqdm  # For progress bar

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# Define the class ID for "sports ball" in the COCO dataset
BALL_CLASS_ID = 32

def count_ball_detections(input_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Check if video is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {input_video_path}")
    print(f"Total frames: {total_frames}")

    # Initialize ball detection counter
    total_ball_detections = 0

    # Process video frame by frame with a progress bar
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Run YOLOv5 model on the current frame
            results = model(frame)

            # Extract bounding boxes and filter for "sports ball"
            detections = results.pred[0]
            ball_count_in_frame = 0  # Track detections per frame

            for *xyxy, conf, cls in detections:
                if int(cls) == BALL_CLASS_ID:  # Filter for ball
                    ball_count_in_frame += 1  # Increment ball count for this frame

            # Update the total count
            total_ball_detections += ball_count_in_frame

            # Show the running total of ball detections
            tqdm.write(f"Frame: {pbar.n+1}/{total_frames} | Balls detected so far: {total_ball_detections}")

            # Update progress bar
            pbar.update(1)

    # Release video objects
    cap.release()
    print(f"Total ball detections in video: {total_ball_detections}")

# Input video path
input_video = './video_rightlong.mp4'  # Input video in the current directory

# Run the function
count_ball_detections(input_video)

import cv2
import numpy as np
from tqdm import tqdm

# Path to input video
input_video_path = "input_video.mp4"
output_video_path = "processed_video2.mp4"

# Load the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Video writer to save the processed output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process video with tqdm progress bar
for _ in tqdm(range(frame_count), desc="Processing video frames"):
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Apply some morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Create a masked frame
    moving_objects = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # Write the processed frame to the output video
    out.write(moving_objects)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_video_path}")

import cv2
import numpy as np
from tqdm import tqdm

def create_motion_video(input_video, output_video, threshold=30, frame_diff=1, max_frames=None):
    """
    Create a motion video with thresholding from the input video.

    Parameters:
        input_video (str): Path to the input video file.
        output_video (str): Path to save the output motion video.
        threshold (int): Pixel intensity difference threshold for motion detection.
        frame_diff (int): Frame difference interval for motion calculation.
        max_frames (int): Maximum number of frames to process. If None, process all frames.

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Limit frames to process if max_frames is specified
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        out.release()
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Process each frame with a progress bar
    frame_count = 0
    with tqdm(total=frames_to_process, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break

            frame_count += 1

            # Skip frames based on frame_diff
            if frame_count % frame_diff != 0:
                pbar.update(1)
                continue

            # Convert the current frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute the absolute difference between frames
            diff = cv2.absdiff(prev_gray, gray)

            # Apply thresholding to the difference image
            _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

            # Write the motion frame to the output video
            out.write(motion_mask)

            # Update the previous frame
            prev_gray = gray

            # Update progress bar
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()
    print(f"Motion video saved as {output_video}")

# Define input and output paths
input_video_path = "video_leftlongshifted.mp4"
output_video_path = "motion_video.mp4"

# Create the motion video
create_motion_video(input_video_path, output_video_path, threshold=30, frame_diff=1, max_frames=300)

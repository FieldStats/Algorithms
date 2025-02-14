import cv2
import numpy as np
from tqdm import tqdm

def process_and_save_frames(input_video, threshold=30, frame_diff=1, save_interval=10, output_folder="frames_output"):
    """
    Process a video with thresholding and save specific frames as images.

    Parameters:
        input_video (str): Path to the input video file.
        threshold (int): Pixel intensity difference threshold for motion detection.
        frame_diff (int): Frame difference interval for motion calculation.
        save_interval (int): Save every nth processed frame.
        output_folder (str): Directory to save the extracted frames.

    Returns:
        None
    """
    import os

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Process each frame with a progress bar
    frame_count = 0
    save_count = 0
    with tqdm(total=total_frames, desc="Processing and Saving Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
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

            # Save the frame if it is the nth frame
            if frame_count % save_interval == 0:
                save_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
                cv2.imwrite(save_path, motion_mask)
                save_count += 1

            # Update the previous frame
            prev_gray = gray

            # Update progress bar
            pbar.update(1)

    # Release resources
    cap.release()
    print(f"Saved {save_count} frames to {output_folder}")

# Define input video path and parameters
input_video_path = "video_leftlongshifted.mp4"
output_folder_path = "extracted_frames"

# Process the video and save frames
process_and_save_frames(
    input_video=input_video_path,
    threshold=30,
    frame_diff=1,
    save_interval=10,
    output_folder=output_folder_path
)

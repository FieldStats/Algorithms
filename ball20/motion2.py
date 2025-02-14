import cv2
import numpy as np
from tqdm import tqdm

def process_video_with_blur_and_threshold(input_video, output_video, blur_kernel=(5, 5), threshold_value=127):
    """
    Apply Gaussian blur and thresholding to a video.

    Parameters:
        input_video (str): Path to the input video file.
        output_video (str): Path to save the processed video.
        blur_kernel (tuple): Size of the kernel for Gaussian blurring.
        threshold_value (int): Threshold value for binary thresholding.

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

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Process each frame with a progress bar
    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure the frame is in grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray_frame, blur_kernel, 0)

            # Apply binary thresholding
            _, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

            # Write the processed frame to the output video
            out.write(thresholded)

            # Update progress bar
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved as {output_video}")

# Define input and output paths
input_video_path = "motion_video.mp4"
output_video_path = "processed_motion_blur_threshold_video.mp4"

# Apply Gaussian blur and thresholding
process_video_with_blur_and_threshold(
    input_video=input_video_path,
    output_video=output_video_path,
    blur_kernel=(7, 7),
    threshold_value=100
)

import cv2
import numpy as np
import os
import threading
from tqdm import tqdm  # Import tqdm for progress bar

# File paths
VIDEO_LEFT = "left5shifted.mp4"
VIDEO_RIGHT = "right5.mp4"
HOMOGRAPHY_MATRIX_LEFT = "al2_homography_matrix.txt"
HOMOGRAPHY_MATRIX_RIGHT = "al1_homography_matrix.txt"
DIMENSIONS_FILE = "dimensions.txt"
OUTPUT_LEFT_VIDEO = "transformed_left_output2.mp4"
OUTPUT_RIGHT_VIDEO = "transformed_right_output2.mp4"
OUTPUT_MERGED_VIDEO = "transformed_merged_output.mp4"

def adjust_blue_lines(blue_line_left, blue_line_right, frame_width):
    """
    Adjust the blue line positions by moving them dynamically.
    - Left video: Move 5% to the right.
    - Right video: Move 2% to the left.
    """
    new_blue_line_left = int(blue_line_left + 0.02 * frame_width)
    new_blue_line_right = int(blue_line_right - 0.01 * frame_width)
    return new_blue_line_left, new_blue_line_right

def process_video(video_file, homography_matrix, output_file, frame_width, frame_height):
    """
    Process a single video and save the transformed output.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_file}'.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Process each frame with a progress bar
    for frame_index in tqdm(range(total_frames), desc=f"Processing {output_file}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Transform the frame
        transformed_frame = cv2.warpPerspective(frame, homography_matrix, (frame_width, frame_height))

        # Write the transformed frame to the output video
        out.write(transformed_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"\nProcessing complete. Output saved as {output_file}")

def merge_videos_with_adjusted_blue_lines(output_file, left_video, right_video, frame_width, frame_height, blue_line_left, blue_line_right):
    """
    Merge left and right videos through dynamically adjusted blue lines.
    """
    cap_left = cv2.VideoCapture(left_video)
    cap_right = cv2.VideoCapture(right_video)

    # Adjust blue lines dynamically
    adjusted_blue_line_left, adjusted_blue_line_right = adjust_blue_lines(blue_line_left, blue_line_right, frame_width)

    # Get properties
    fps = int(cap_left.get(cv2.CAP_PROP_FPS))
    total_frames = int(min(cap_left.get(cv2.CAP_PROP_FRAME_COUNT), cap_right.get(cv2.CAP_PROP_FRAME_COUNT)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_width = adjusted_blue_line_left + (frame_width - adjusted_blue_line_right)
    out = cv2.VideoWriter(output_file, fourcc, fps, (output_width, frame_height))

    for _ in tqdm(range(total_frames), desc="Merging videos"):
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        if not ret_left or not ret_right:
            break

        # Crop frames based on adjusted blue line
        left_cropped = frame_left[:, :adjusted_blue_line_left]
        right_cropped = frame_right[:, adjusted_blue_line_right:]

        # Concatenate frames at the blue line
        combined_frame = np.hstack((left_cropped, right_cropped))
        out.write(combined_frame)

    cap_left.release()
    cap_right.release()
    out.release()
    print(f"Merged video through adjusted blue lines saved as {output_file}")

def main():
    # Check if files exist
    required_files = [
        VIDEO_LEFT, VIDEO_RIGHT, HOMOGRAPHY_MATRIX_LEFT, HOMOGRAPHY_MATRIX_RIGHT, DIMENSIONS_FILE
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return

    # Load dimensions and homography matrices
    with open(DIMENSIONS_FILE, "r") as f:
        lines = f.readlines()
    blue_line_right, _ = map(int, lines[0].split())
    blue_line_left, _ = map(int, lines[1].split())

    homography_matrix_left = np.loadtxt(HOMOGRAPHY_MATRIX_LEFT, delimiter=' ')
    homography_matrix_right = np.loadtxt(HOMOGRAPHY_MATRIX_RIGHT, delimiter=' ')

    # Transformed image dimensions
    frame_width = 400  # Adjust as needed
    frame_height = 300  # Adjust as needed

    # Create threads for video processing
    left_thread = threading.Thread(target=process_video, args=(VIDEO_LEFT, homography_matrix_left, OUTPUT_LEFT_VIDEO, frame_width, frame_height))
    right_thread = threading.Thread(target=process_video, args=(VIDEO_RIGHT, homography_matrix_right, OUTPUT_RIGHT_VIDEO, frame_width, frame_height))

    # Start both threads
    left_thread.start()
    right_thread.start()

    # Wait for both threads to complete
    left_thread.join()
    right_thread.join()

    # Merge the processed videos
    merge_videos_with_adjusted_blue_lines(OUTPUT_MERGED_VIDEO, OUTPUT_LEFT_VIDEO, OUTPUT_RIGHT_VIDEO, frame_width, frame_height, blue_line_left, blue_line_right)

    import unifyforbytetrack
    print("fourth script completed. Now running the fifth script...")
    unifyforbytetrack.main()  # Call the second script after finishing the first


if __name__ == "__main__":
    main()
    

    
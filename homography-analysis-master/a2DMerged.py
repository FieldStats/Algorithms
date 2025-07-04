import cv2
import numpy as np
import json
import os
from tqdm import tqdm  # Import tqdm
import threading

# File paths
JSON_LEFT_INTERSECTION = "new_left_intersections.json"
JSON_LEFT_NON_INTERSECTION = "left_non_intersections.json"
JSON_RIGHT_INTERSECTION = "new_right_intersections.json"
JSON_RIGHT_NON_INTERSECTION = "right_non_intersections.json"
VIDEO_LEFT = "left5shifted.mp4"
VIDEO_RIGHT = "right5.mp4"
HOMOGRAPHY_MATRIX_LEFT = "al2_homography_matrix.txt"
HOMOGRAPHY_MATRIX_RIGHT = "al1_homography_matrix.txt"
DIMENSIONS_FILE = "dimensions.txt"
OUTPUT_LEFT_VIDEO = "transformed_left_output2.mp4"
OUTPUT_RIGHT_VIDEO = "transformed_right_output2.mp4"
OUTPUT_MERGED_VIDEO = "transformed_merged_output.mp4"

# Updated Color mapping
COLOR_MAP = {
    "blue": (255, 0, 0),      # Non-intersection left
    "red": (0, 0, 255),       # Non-intersection right
    "purple": (128, 0, 128),  # Intersection left
    "orange": (0, 165, 255),  # Intersection right
    "yellow": (0, 255, 255),  # Derived from orange
    "pink": (255, 20, 147),   # Derived from purple
    "unknown": (255, 255, 255)  # Default unknown color
}

def adjust_blue_lines(blue_line_left, blue_line_right, frame_width):
    """
    Adjust the blue line positions by moving them dynamically.
    - Left video: Move 5% to the right.
    - Right video: Move 2% to the left.
    """
    new_blue_line_left = int(blue_line_left + 0.01 * frame_width)
    new_blue_line_right = int(blue_line_right - 0.005 * frame_width)
    return new_blue_line_left, new_blue_line_right

def draw_objects_as_dots_with_ids(frame, objects, homography_matrix):
    """
    Draw the center points of objects as dots with their IDs on the transformed frame.
    """
    for obj in objects:
        track_id = obj.get("track_id")
        center = obj.get("center")
        color_name = obj.get("color", "unknown")  # Default
        color = COLOR_MAP.get(color_name, (255, 255, 255))  # Use default color if not mapped

        # Convert center to a numpy array
        center_point = np.array([[center]], dtype=np.float32)

        # Transform the center point using the homography matrix
        transformed_center = cv2.perspectiveTransform(center_point, homography_matrix)

        # Draw the transformed center point as a dot
        transformed_center = tuple(transformed_center[0][0].astype(int))
        cv2.circle(frame, transformed_center, 5, color, -1)  # Dot with specified color

        # Draw the ID near the dot
        cv2.putText(frame, f"ID: {track_id}", (transformed_center[0] + 10, transformed_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

    return frame

def process_video(video_file, json_intersection, json_non_intersection, homography_matrix, output_file, frame_width, frame_height):
    """
    Process a single video with its intersection and non-intersection JSON data and save the transformed output.
    """
    # Load JSON data
    with open(json_intersection, "r") as f:
        data_intersection = json.load(f)
    with open(json_non_intersection, "r") as f:
        data_non_intersection = json.load(f)

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

        # Get objects for the current frame
        frame_data_intersection = next((item for item in data_intersection if item["frame_index"] == frame_index), None)
        frame_data_non_intersection = next((item for item in data_non_intersection if item["frame_index"] == frame_index), None)

        if frame_data_intersection:
            objects = frame_data_intersection.get("objects", [])
            transformed_frame = draw_objects_as_dots_with_ids(transformed_frame, objects, homography_matrix)

        if frame_data_non_intersection:
            objects = frame_data_non_intersection.get("objects", [])
            transformed_frame = draw_objects_as_dots_with_ids(transformed_frame, objects, homography_matrix)

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

    # Merge frames with a progress bar
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
        JSON_LEFT_INTERSECTION, JSON_LEFT_NON_INTERSECTION, JSON_RIGHT_INTERSECTION, JSON_RIGHT_NON_INTERSECTION,
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

    # Process left video
    left_thread = threading.Thread(target=process_video, args=(VIDEO_LEFT, JSON_LEFT_INTERSECTION, JSON_LEFT_NON_INTERSECTION, homography_matrix_left, OUTPUT_LEFT_VIDEO, frame_width, frame_height))
    right_thread = threading.Thread(target=process_video, args=(VIDEO_RIGHT, JSON_RIGHT_INTERSECTION, JSON_RIGHT_NON_INTERSECTION, homography_matrix_right, OUTPUT_RIGHT_VIDEO, frame_width, frame_height))

    # Start both threads
    left_thread.start()
    right_thread.start()

    # Wait for both threads to complete
    left_thread.join()
    right_thread.join()
    # Merge the processed videos
    merge_videos_with_adjusted_blue_lines(OUTPUT_MERGED_VIDEO, OUTPUT_LEFT_VIDEO, OUTPUT_RIGHT_VIDEO,
                                            frame_width, frame_height, blue_line_left, blue_line_right)

if __name__ == "__main__":
    main()

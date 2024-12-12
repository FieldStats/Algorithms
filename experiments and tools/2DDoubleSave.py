import cv2
import numpy as np
import json
import os

# File paths
JSON_LEFT_INTERSECTION = "new_left_intersections.json"
JSON_LEFT_NON_INTERSECTION = "left_non_intersections.json"
JSON_RIGHT_INTERSECTION = "new_right_intersections.json"
JSON_RIGHT_NON_INTERSECTION = "right_non_intersections.json"
VIDEO_LEFT = "video_leftlongshifted.mp4"
VIDEO_RIGHT = "video_rightlong.mp4"
HOMOGRAPHY_MATRIX_LEFT = "al2_homography_matrix.txt"
HOMOGRAPHY_MATRIX_RIGHT = "al1_homography_matrix.txt"
OUTPUT_LEFT_VIDEO = "transformed_left_output2.mp4"
OUTPUT_RIGHT_VIDEO = "transformed_right_output2.mp4"

# Updated Color mapping
COLOR_MAP = {
    "blue": (255, 0, 0),      # Non-intersection left
    "red": (0, 0, 255),       # Non-intersection right
    "purple": (128, 0, 128),  # Intersection left
    "orange": (0, 165, 255),  # Intersection right
    "yellow": (0, 255, 255),  # New: Derived from orange
    "pink": (255, 20, 147),   # New: Derived from purple
    "unknown": (255, 255, 255)  # Default unknown color
}

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

    frame_index = 0
    while frame_index < total_frames:
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

        # Show progress
        print(f"Processing frame {frame_index + 1}/{total_frames} for {output_file}...", end="\r")
        frame_index += 1

    # Release resources
    cap.release()
    out.release()
    print(f"\nProcessing complete. Output saved as {output_file}")


def main():
    # Check if files exist
    required_files = [
        JSON_LEFT_INTERSECTION, JSON_LEFT_NON_INTERSECTION, JSON_RIGHT_INTERSECTION, JSON_RIGHT_NON_INTERSECTION,
        VIDEO_LEFT, VIDEO_RIGHT, HOMOGRAPHY_MATRIX_LEFT, HOMOGRAPHY_MATRIX_RIGHT
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return

    # Load homography matrices
    homography_matrix_left = np.loadtxt(HOMOGRAPHY_MATRIX_LEFT, delimiter=' ')
    homography_matrix_right = np.loadtxt(HOMOGRAPHY_MATRIX_RIGHT, delimiter=' ')

    # Transformed image dimensions
    frame_width = 400  # Adjust as needed
    frame_height = 300  # Adjust as needed

    # Process left video
    process_video(VIDEO_LEFT, JSON_LEFT_INTERSECTION, JSON_LEFT_NON_INTERSECTION, homography_matrix_left, OUTPUT_LEFT_VIDEO, frame_width, frame_height)

    # Process right video
    process_video(VIDEO_RIGHT, JSON_RIGHT_INTERSECTION, JSON_RIGHT_NON_INTERSECTION, homography_matrix_right, OUTPUT_RIGHT_VIDEO, frame_width, frame_height)


if __name__ == "__main__":
    main()

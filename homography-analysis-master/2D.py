import cv2
import numpy as np
import json
import os

# File paths
JSON_FILE = "rightmerged.json"
VIDEO_FILE = "video_rightlong.mp4"
HOMOGRAPHY_MATRIX_FILE = "al1_homography_matrix.txt"
OUTPUT_VIDEO_FILE = "transformed_output_video_with_dots_and_ids.mp4"

# Color mapping
COLOR_MAP = {
    "blue": (255, 0, 0),     # BGR format
    "red": (0, 0, 255),
    "purple": (128, 0, 128),
    "orange": (0, 165, 255)
}

def draw_objects_as_dots_with_ids(frame, objects, homography_matrix):
    """
    Draw the center points of objects as dots with their IDs on the transformed frame.
    """
    for obj in objects:
        track_id = obj.get("track_id")
        center = obj.get("center")
        color_name = obj.get("color", "red")  # Default to red if no color specified
        color = COLOR_MAP.get(color_name, (0, 0, 255))  # Default to red if unknown color

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

def main():
    # Check if files exist
    if not os.path.exists(JSON_FILE):
        print(f"Error: JSON file '{JSON_FILE}' not found.")
        return
    if not os.path.exists(VIDEO_FILE):
        print(f"Error: Video file '{VIDEO_FILE}' not found.")
        return
    if not os.path.exists(HOMOGRAPHY_MATRIX_FILE):
        print(f"Error: Homography matrix file '{HOMOGRAPHY_MATRIX_FILE}' not found.")
        return

    # Load JSON data
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    # Load the homography matrix
    homography_matrix = np.loadtxt(HOMOGRAPHY_MATRIX_FILE, delimiter=' ')

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{VIDEO_FILE}'.")
        return

    # Get video properties
    frame_width = 400  # Transformed frame width
    frame_height = 300  # Transformed frame height
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Transform the frame
        transformed_frame = cv2.warpPerspective(frame, homography_matrix, (frame_width, frame_height))

        # Get objects for the current frame
        frame_data = next((item for item in data if item["frame_index"] == frame_index), None)
        if frame_data:
            objects = frame_data.get("objects", [])
            transformed_frame = draw_objects_as_dots_with_ids(transformed_frame, objects, homography_matrix)

        # Write the transformed frame to the output video
        out.write(transformed_frame)

        # Show progress
        print(f"Processing frame {frame_index + 1}/{total_frames}...", end="\r")
        frame_index += 1

    # Release resources
    cap.release()
    out.release()
    print("\nProcessing complete. Output saved as", OUTPUT_VIDEO_FILE)

if __name__ == "__main__":
    main()

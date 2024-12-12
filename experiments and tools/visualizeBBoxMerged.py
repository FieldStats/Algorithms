import cv2
import numpy as np
import json
import os

# File paths
JSON_FILE = "modified_tracking_data.json"
VIDEO_LEFT = "video_left.mp4"
VIDEO_RIGHT = "video_left.mp4"
HOMOGRAPHY_MATRIX_LEFT = "al2_homography_matrix.txt"
HOMOGRAPHY_MATRIX_RIGHT = "al2_homography_matrix.txt"
OUTPUT_VIDEO_FILE = "transformed_side_by_side_output.mp4"

def draw_objects_as_dots_with_ids(frame, objects, homography_matrix):
    """
    Draw the middle-bottom points of bounding boxes as dots with their IDs on the transformed frame.
    """
    for obj in objects:
        track_id = obj.get("track_id")
        bbox = obj.get("bbox")  # Bounding box: [x_min, y_min, x_max, y_max]

        # Calculate the middle-bottom point of the bounding box
        middle_bottom = [(bbox[0] + bbox[2]) / 2, bbox[3]]  # (middle of bottom edge)

        # Convert middle-bottom to a numpy array
        bottom_point = np.array([[middle_bottom]], dtype=np.float32)

        # Transform the middle-bottom point using the homography matrix
        transformed_bottom = cv2.perspectiveTransform(bottom_point, homography_matrix)

        # Draw the transformed middle-bottom point as a dot
        transformed_bottom = tuple(transformed_bottom[0][0].astype(int))
        cv2.circle(frame, transformed_bottom, 5, (0, 0, 255), -1)  # Red dot

        # Draw the ID near the dot
        cv2.putText(frame, f"ID: {track_id}", (transformed_bottom[0] + 10, transformed_bottom[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def process_frame(cap, homography_matrix, frame_width, frame_height, frame_index, data):
    """
    Process a single frame, transform it using the homography matrix, and annotate objects.
    """
    ret, frame = cap.read()
    if not ret:
        return None

    # Transform the frame
    transformed_frame = cv2.warpPerspective(frame, homography_matrix, (frame_width, frame_height))

    # Get objects for the current frame
    frame_data = next((item for item in data if item["frame_index"] == frame_index), None)
    if frame_data:
        objects = frame_data.get("objects", [])
        transformed_frame = draw_objects_as_dots_with_ids(transformed_frame, objects, homography_matrix)

    return transformed_frame

def main():
    # Check if files exist
    if not os.path.exists(JSON_FILE):
        print(f"Error: JSON file '{JSON_FILE}' not found.")
        return
    if not os.path.exists(VIDEO_LEFT):
        print(f"Error: Video file '{VIDEO_LEFT}' not found.")
        return
    if not os.path.exists(VIDEO_RIGHT):
        print(f"Error: Video file '{VIDEO_RIGHT}' not found.")
        return
    if not os.path.exists(HOMOGRAPHY_MATRIX_LEFT) or not os.path.exists(HOMOGRAPHY_MATRIX_RIGHT):
        print("Error: One or both homography matrix files not found.")
        return

    # Load JSON data
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    # Load homography matrices
    homography_matrix_left = np.loadtxt(HOMOGRAPHY_MATRIX_LEFT, delimiter=' ')
    homography_matrix_right = np.loadtxt(HOMOGRAPHY_MATRIX_RIGHT, delimiter=' ')

    # Open video files
    cap_left = cv2.VideoCapture(VIDEO_LEFT)
    cap_right = cv2.VideoCapture(VIDEO_RIGHT)
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Cannot open one or both video files.")
        return

    # Get video properties
    frame_width = 400  # Transformed frame width for each camera
    frame_height = 300  # Transformed frame height for each camera
    fps = int(cap_left.get(cv2.CAP_PROP_FPS))
    total_frames = int(min(cap_left.get(cv2.CAP_PROP_FRAME_COUNT), cap_right.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, fps, (frame_width * 2, frame_height))

    frame_index = 0
    while frame_index < total_frames:
        print(f"Processing frame {frame_index + 1}/{total_frames}...", end="\r")

        # Process frames for both cameras
        frame_left = process_frame(cap_left, homography_matrix_left, frame_width, frame_height, frame_index, data)
        frame_right = process_frame(cap_right, homography_matrix_right, frame_width, frame_height, frame_index, data)

        if frame_left is None or frame_right is None:
            break

        # Adjoin frames side by side
        combined_frame = np.hstack((frame_left, frame_right))

        # Write the combined frame to the output video
        out.write(combined_frame)

        frame_index += 1

    # Release resources
    cap_left.release()
    cap_right.release()
    out.release()
    print("\nProcessing complete. Output saved as", OUTPUT_VIDEO_FILE)

if __name__ == "__main__":
    main()

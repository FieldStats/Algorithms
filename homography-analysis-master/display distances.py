import cv2
import numpy as np
import json
import os
from collections import defaultdict

# File paths
JSON_FILE = "modified_tracking_data.json"
VIDEO_FILE = "new_test2.mp4"
HOMOGRAPHY_MATRIX_FILE = "homography_matrix.txt"
OUTPUT_VIDEO_FILE = "transformed_video_with_bottom_middle_interpolation.mp4"

# Real-world dimensions of the transformed image
REAL_WIDTH = 100  # Real-world width of the transformed image
REAL_HEIGHT = 75  # Real-world height of the transformed image

# Track the last known positions for interpolation
last_positions = {}
last_frame_indices = {}

def calculate_bottom_middle(bbox):
    """
    Calculate the bottom-middle point of the bounding box.
    """
    x_bottom_middle = (bbox[0] + bbox[2]) / 2
    y_bottom_middle = bbox[3]
    return np.array([x_bottom_middle, y_bottom_middle])

def calculate_real_distance(p1, p2, image_width, image_height):
    """
    Calculate the real-world distance between two points using the transformed image size.
    """
    dx = (p2[0] - p1[0]) * (REAL_WIDTH / image_width)
    dy = (p2[1] - p1[1]) * (REAL_HEIGHT / image_height)
    return np.sqrt(dx**2 + dy**2)

def interpolate_position(track_id, frame_index, last_known_position, last_known_frame, current_frame):
    """
    Linearly interpolate the position of an object.
    """
    if track_id not in last_positions:
        return last_known_position  # No data to interpolate from
    
    t1 = last_known_frame
    t2 = current_frame
    t = frame_index

    p1 = np.array(last_positions[track_id])
    p2 = np.array(last_known_position)

    interpolated_position = p1 + (p2 - p1) * ((t - t1) / (t2 - t1))
    return interpolated_position

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

    # Track distances covered by each object
    distances = defaultdict(float)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Transform the frame
        transformed_frame = cv2.warpPerspective(frame, homography_matrix, (frame_width, frame_height))

        # Get objects for the current frame
        frame_data = next((item for item in data if item["frame_index"] == frame_index), None)
        detected_objects = {}

        if frame_data:
            for obj in frame_data["objects"]:
                track_id = obj["track_id"]
                bbox = obj["bbox"]

                # Calculate the bottom-middle point of the bounding box
                bottom_middle = calculate_bottom_middle(bbox)

                # Transform the bottom-middle point using the homography matrix
                bottom_middle_point = np.array([[bottom_middle]], dtype=np.float32)
                transformed_bottom_middle = cv2.perspectiveTransform(bottom_middle_point, homography_matrix)[0][0]

                # Store the detected position
                detected_objects[track_id] = transformed_bottom_middle

                # Update position and frame information
                last_positions[track_id] = transformed_bottom_middle
                last_frame_indices[track_id] = frame_index

                # Draw the transformed bottom-middle point as a red dot
                transformed_bottom_middle = tuple(transformed_bottom_middle.astype(int))
                cv2.circle(transformed_frame, transformed_bottom_middle, 5, (0, 0, 255), -1)  # Red dot
                cv2.putText(transformed_frame, f"ID: {track_id}", (transformed_bottom_middle[0] + 10, transformed_bottom_middle[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Update distances
                if track_id in last_positions and frame_index - last_frame_indices[track_id] == 1:
                    distances[track_id] += calculate_real_distance(last_positions[track_id], transformed_bottom_middle, frame_width, frame_height)

        # Interpolate missing objects
        for track_id in last_positions.keys():
            if track_id not in detected_objects:
                interpolated_position = interpolate_position(
                    track_id,
                    frame_index,
                    last_positions[track_id],
                    last_frame_indices[track_id],
                    frame_index + 1
                )
                # Draw interpolated position as a purple dot
                interpolated_position = tuple(interpolated_position.astype(int))
                cv2.circle(transformed_frame, interpolated_position, 5, (255, 0, 255), -1)  # Purple dot
                cv2.putText(transformed_frame, f"ID: {track_id}", (interpolated_position[0] + 10, interpolated_position[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Update distances
                if track_id in last_positions:
                    distances[track_id] += calculate_real_distance(last_positions[track_id], interpolated_position, frame_width, frame_height)

                # Update last position to the interpolated position
                last_positions[track_id] = interpolated_position

        # Display cumulative distances
        y_offset = 20
        for track_id, distance in distances.items():
            text = f"ID: {track_id}, Distance: {distance:.2f} units"
            cv2.putText(transformed_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15

        # Write the frame to the output video
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

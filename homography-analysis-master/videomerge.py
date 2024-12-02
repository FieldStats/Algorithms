import cv2
import json
import os

# File paths
JSON_FILE = "modified_tracking_data.json"
VIDEO_FILE = "new_test2.mp4"
OUTPUT_VIDEO_FILE = "output_video.mp4"

def draw_objects_on_frame(frame, objects):
    """
    Draw bounding boxes, centers, and other details on the video frame.
    """
    for obj in objects:
        track_id = obj.get("track_id")
        class_id = obj.get("class_id")
        confidence = obj.get("confidence")
        bbox = obj.get("bbox")
        center = obj.get("center")

        # Convert bounding box and center coordinates to integers
        bbox = [int(coord) for coord in bbox]
        center = [int(coord) for coord in center]

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Draw center point
        cv2.circle(frame, (center[0], center[1]), 5, (0, 0, 255), -1)

        # Add text for track ID, class ID, and confidence
        text = f"ID: {track_id}, Class: {class_id}, Conf: {confidence:.2f}"
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

def main():
    # Check if files exist
    if not os.path.exists(JSON_FILE):
        print(f"Error: JSON file '{JSON_FILE}' not found.")
        return
    if not os.path.exists(VIDEO_FILE):
        print(f"Error: Video file '{VIDEO_FILE}' not found.")
        return

    # Load JSON data
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{VIDEO_FILE}'.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

        # Get objects for the current frame
        frame_data = next((item for item in data if item["frame_index"] == frame_index), None)
        if frame_data:
            objects = frame_data.get("objects", [])
            frame = draw_objects_on_frame(frame, objects)

        # Write the frame to the output video
        out.write(frame)

        # Show progress
        print(f"Processing frame {frame_index + 1}/{total_frames}...", end="\r")
        frame_index += 1

    # Release resources
    cap.release()
    out.release()
    print("\nProcessing complete. Output saved as", OUTPUT_VIDEO_FILE)

if __name__ == "__main__":
    main()

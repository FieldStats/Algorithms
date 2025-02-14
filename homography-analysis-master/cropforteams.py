import os
import cv2
import json

# Paths to input JSON files and video files
TRACKING_DATA_RIGHT = "righttrack.json"
TRACKING_DATA_LEFT = "lefttrack.json"
VIDEO_RIGHT = "video_rightlong.mp4"
VIDEO_LEFT = "video_leftlongshifted.mp4"

# Output directories
OUTPUT_DIR_RIGHT = "cropped_objects_right"
OUTPUT_DIR_LEFT = "cropped_objects_left"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR_RIGHT, exist_ok=True)
os.makedirs(OUTPUT_DIR_LEFT, exist_ok=True)

# Function to crop and save the first 100 objects from a JSON and video
def crop_and_save_objects(json_path, video_path, output_dir, max_objects=100):
    with open(json_path, "r") as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    count = 0  # Counter for cropped objects
    for frame_data in tracking_data:
        if count >= max_objects:
            break

        frame_index = frame_data["frame_index"]
        objects = frame_data["objects"]

        # Set video frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frame {frame_index} from video")
            continue

        for obj in objects:
            if count >= max_objects:
                break

            # Get bounding box and crop ROI
            bbox = obj["bbox"]
            x_min, y_min, x_max, y_max = map(int, bbox)
            roi = frame[y_min:y_max, x_min:x_max]

            # Skip invalid ROIs
            if roi.size == 0:
                continue

            # Save the cropped ROI
            output_path = os.path.join(output_dir, f"object_{count}.jpg")
            cv2.imwrite(output_path, roi)
            print(f"Saved: {output_path}")
            count += 1

    cap.release()
    print(f"Saved {count} objects to {output_dir}")

# Process both JSONs and videos
crop_and_save_objects(TRACKING_DATA_RIGHT, VIDEO_RIGHT, OUTPUT_DIR_RIGHT)
crop_and_save_objects(TRACKING_DATA_LEFT, VIDEO_LEFT, OUTPUT_DIR_LEFT)

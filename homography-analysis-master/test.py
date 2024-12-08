import os
import json
import cv2
from tqdm import tqdm

# File paths in the current directory
JSON_FILE = "tracking_data_1.json"
VIDEO_FILE = "video_rightlong.mp4"
OUTPUT_VIDEO_FILE = "transformed_output_video_with_dots_and_ids.mp4"


def overlay_tracking_data_on_video(json_file, video_file, output_video_file, confidence_threshold=0.1):
    """
    Overlay tracking data on video frames and save the transformed video.

    :param json_file: JSON file with tracking data in the current directory.
    :param video_file: Video file in the current directory.
    :param output_video_file: Name of the output video file in the current directory.
    :param confidence_threshold: Minimum confidence score to display objects (default 0.1).
    """
    # Load tracking data from JSON
    with open(json_file, 'r') as json_data:
        tracking_data = json.load(json_data)

    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # Process each frame with a progress bar
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                break

            # Get tracking data for the current frame
            if frame_idx < len(tracking_data):
                frame_info = tracking_data[frame_idx]

                for obj in frame_info.get("objects", []):
                    class_id = obj["class_id"]
                    center = obj["center"]
                    track_id = obj.get("track_id", "N/A")
                    confidence = obj["confidence"]

                    # Skip low-confidence detections
                    if confidence < confidence_threshold:
                        continue

                    # Draw bounding box center
                    cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)

                    # Annotate track ID near the detection point
                    cv2.putText(frame, f"ID: {track_id}", 
                                (int(center[0]) + 10, int(center[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Write the frame to the output video
            out.write(frame)
            pbar.update(1)  # Update the progress bar

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved to {output_video_file}")


if __name__ == "__main__":
    # Overlay tracking data on the video and save the output
    overlay_tracking_data_on_video(JSON_FILE, VIDEO_FILE, OUTPUT_VIDEO_FILE)

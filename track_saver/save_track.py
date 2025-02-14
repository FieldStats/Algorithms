import os
import sys
import json
from ultralytics import YOLO
import supervision as sv  # Includes ByteTrack implementation


def save_tracking_data_with_bytetrack(video_path, model_path, output_json, device):
    """
    Perform object detection and tracking on a video and save results to a JSON file.

    :param video_path: Path to the input video.
    :param model_path: Path to the YOLO model weights.
    :param output_json: Path to save the output JSON with tracking data.
    :param device: Device to use for inference ("cpu" or "gpu").
    """
    # Initialize YOLO model
    model = YOLO(model_path)

    # Initialize ByteTrack
    tracker = sv.ByteTrack()

    # Run prediction
    device_option = 0 if device == "gpu" else "cpu"
    results = model.predict(
    video_path, 
    verbose=True, 
    save=False, 
    save_txt=False, 
    save_conf=False, 
    device=device_option, 
    imgsz=1080  # Specify the desired resolution (e.g., 640)
)


    # Prepare tracking data
    tracking_data = []  # List to hold frame-wise tracking information

    for frame_idx, result in enumerate(results):
        frame_data = {"frame_index": frame_idx, "objects": []}

        # Convert YOLO detections to Supervision's Detections format
        detection_supervision = sv.Detections.from_ultralytics(result)

        # Update the tracker with detections
        tracked_objects = tracker.update_with_detections(detection_supervision)

        # Loop through tracked objects to extract relevant information
        for track in tracked_objects:
            bbox = track[0].tolist()  # Bounding box [x1, y1, x2, y2]
            track_id = track[4]       # Unique track ID
            cls_id = track[3]         # Class ID
            confidence = track[2]     # Confidence score
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Append object data with track ID
            frame_data["objects"].append({
                "track_id": int(track_id),
                "class_id": int(cls_id),
                "confidence": float(confidence),
                "bbox": list(map(float, bbox)),
                "center": list(map(float, [(center_x), (center_y)])),
            })

        tracking_data.append(frame_data)

    # Save tracking data to JSON
    with open(output_json, 'w') as json_file:
        json.dump(tracking_data, json_file, indent=4)

    print(f"Tracking data with ByteTrack saved to {output_json}")


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print(f"Usage: python {sys.argv[0]} <videoname> <modelname> <outputjson> [device]")
        sys.exit(1)

    videoname = sys.argv[1]
    modelname = sys.argv[2]
    outputjson = sys.argv[3]
    device = sys.argv[4] if len(sys.argv) == 5 else "cpu"

    video_path = os.path.join('input_videos', videoname)
    model_path = os.path.join('models', modelname)
    output_json = os.path.join('json_output', outputjson)

    save_tracking_data_with_bytetrack(video_path, model_path, output_json, device)

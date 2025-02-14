import os
import sys
import json
from ultralytics import YOLO

def save_yolo_detections(video_path, model_path, output_json, device):
    """
    Perform object detection on a video using YOLO and save results to a JSON file.

    :param video_path: Path to the input video.
    :param model_path: Path to the YOLO model weights.
    :param output_json: Path to save the output JSON with detection data.
    :param device: Device to use for inference ("cpu" or "gpu").
    """
    # Initialize YOLO model
    model = YOLO(model_path)

    # Run prediction
    device_option = 0 if device == "gpu" else "cpu"
    results = model.predict(video_path, verbose=True, save=False, save_txt=False, save_conf=False, device=device_option)

    # Prepare detection data
    detection_data = []  # List to hold frame-wise detection information

    for frame_idx, result in enumerate(results):
        frame_data = {"frame_index": frame_idx, "objects": []}

        # Loop through YOLO detections for the current frame
        for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            bbox = box.tolist()  # Bounding box [x1, y1, x2, y2]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            frame_data["objects"].append({
                "class_id": int(cls_id.item()),
                "confidence": float(conf.item()),
                "bbox": list(map(float, bbox)),
                "center": [float(center_x), float(center_y)]
            })

        detection_data.append(frame_data)

    # Save detection data to JSON
    with open(output_json, 'w') as json_file:
        json.dump(detection_data, json_file, indent=4)

    print(f"YOLO detection data saved to {output_json}")

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

    save_yolo_detections(video_path, model_path, output_json, device)

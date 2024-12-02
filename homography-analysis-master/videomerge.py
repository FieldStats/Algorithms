import json
import os
from collections import defaultdict

def calculate_iou(box1, box2):
    # Compute IoU between two bounding boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def merge_frame_detections(frame1, frame2, iou_threshold=0.5):
    merged_objects = []
    used_indices = set()

    for obj1 in frame1.get("objects", []):
        merged = False
        for idx, obj2 in enumerate(frame2.get("objects", [])):
            if idx in used_indices:
                continue
            iou = calculate_iou(obj1["bbox"], obj2["bbox"])
            if iou > iou_threshold:
                # Merge based on higher confidence
                if obj1["confidence"] > obj2["confidence"]:
                    merged_objects.append(obj1)
                else:
                    merged_objects.append(obj2)
                used_indices.add(idx)
                merged = True
                break
        if not merged:
            merged_objects.append(obj1)

    # Add remaining objects from frame2
    for idx, obj2 in enumerate(frame2.get("objects", [])):
        if idx not in used_indices:
            merged_objects.append(obj2)

    return {"frame_index": frame1["frame_index"], "objects": merged_objects}

def merge_camera_data(file1, file2, output_file, iou_threshold=0.5):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    max_frames = max(len(data1), len(data2))
    merged_data = []

    for i in range(max_frames):
        frame1 = data1[i] if i < len(data1) else {"frame_index": i, "objects": []}
        frame2 = data2[i] if i < len(data2) else {"frame_index": i, "objects": []}
        merged_frame = merge_frame_detections(frame1, frame2, iou_threshold)
        merged_data.append(merged_frame)

    with open(output_file, "w") as out:
        json.dump(merged_data, out, indent=2)

if __name__ == "__main__":
    input_dir = "./"
    file1 = os.path.join(input_dir, "tracking_data1.json")
    file2 = os.path.join(input_dir, "tracking_data2.json")
    output_file = os.path.join(input_dir, "merged_tracking_data.json")

    merge_camera_data(file1, file2, output_file, iou_threshold=0.5)

import json
import os
import sys
import supervision as sv  # Includes ByteTrack implementation
import matplotlib.pyplot as plt  # For plotting

import numpy as np


def perform_tracking_from_json(input_json, output_json):
    """
    Perform tracking using ByteTrack based on bounding box information from an input JSON.

    :param input_json: Path to the input JSON file with frame detections.
    :param output_json: Path to save the output JSON with tracking data.
    :param fixed_class_id: Fixed class_id to assign to all detections (default is 0).
    """
    # Initialize ByteTrack
    tracker = sv.ByteTrack(
        track_activation_threshold=0.1,
        minimum_matching_threshold=0.98,
        lost_track_buffer=10,
        frame_rate=59,
        minimum_consecutive_frames=1
    )

    # Track management
    max_allowed_id = 23      # Maximum ID value allowed
    active_tracks = {}
    reusable_ids = list(range(1, max_allowed_id + 1))  # Pool of IDs to reuse
    track_id_map = {}  # Map external track IDs to internal IDs
    frame_count = 0
    active_track_counts = []

    # Load input JSON
    with open(input_json, 'r') as f:
        input_data = json.load(f)

    # Prepare tracking data
    tracking_data = []

    for frame_data in input_data:

        frame_count += 1
        frame_index = frame_data["frame_index"]
        detections = frame_data["objects"]
        if frame_index < 428:
            tracking_data.append({"frame_index": frame_index, "objects": []})
            continue

        bboxes = []
        confidences = []
        class_ids = []
        for obj in detections:
            if obj["source"] == "right":
                obj["transformed_center"][0] += 347
            bbox = [
                obj["transformed_center"][0] - 2.5,
                obj["transformed_center"][1] - 2.5,
                obj["transformed_center"][0] + 2.5,
                obj["transformed_center"][1] + 2.5
            ]
            bboxes.append(bbox)
            class_ids.append( obj["team_index"] if "team_index" in obj else -1)
            confidences.append(obj["confidence"])

        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)

        detection_supervision = sv.Detections(
            xyxy=bboxes,
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=np.int32)
        )

        tracked_objects = tracker.update_with_detections(detection_supervision)

        frame_tracking_data = {"frame_index": frame_index, "objects": []}


        for track in tracked_objects:

            bbox = track[0].tolist()
            confidence = track[2]
            class_id = track[3]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            external_id = track[4]
            if external_id not in track_id_map:
                if reusable_ids:
                    distances = []
                    for internal_id in reusable_ids:
                        if internal_id in active_tracks:
                           
                            if True: #(class_id == active_tracks[internal_id]["cls_id"]) or (class_id == -1 or active_tracks[internal_id]["cls_id"] == -1):
                                prev_center = active_tracks[internal_id]["center"]
                                distance_delta = np.sqrt((center_x - prev_center[0]) ** 2 + (center_y - prev_center[1]) ** 2) 
                                distances.append((internal_id, distance_delta))
                                # if active_tracks[internal_id]["cls_id"] != -1:
                                #     class_id = active_tracks[internal_id]["cls_id"]
                        else:
                            distances.append((internal_id, 0))

                    if len(distances) == 0:
                        continue
                    # Choose the ID with the minimum distance
                    min_id, min_distance = min(distances, key=lambda x: x[1])
                    
                    # if min_distance > 35: #prev= 28
                    #     continue
                    
                    internal_id = min_id
                    reusable_ids.remove(internal_id)
                   
                    print("Put,", internal_id, ",at,", frame_index, f",with distance={min_distance}")
                    track_id_map[external_id] = min_id

                    active_tracks[internal_id] = {"frame_count": frame_count, "center": [center_x, center_y], "cls_id":class_id, "active": True}
                else:
                    continue
            else:
                internal_id = track_id_map[external_id]
                active_tracks[internal_id]["frame_count"] = frame_count
                active_tracks[internal_id]["center"] = [center_x, center_y]

                # if class_id != active_tracks[internal_id]["cls_id"] and class_id != -1 and active_tracks[internal_id]["cls_id"] != -1: 
                #     active_tracks[internal_id]["active"] = False
                # elif active_tracks[internal_id]["cls_id"] == -1:
                #     active_tracks[internal_id]["cls_id"] = class_id

            frame_tracking_data["objects"].append({
                "track_id": int(internal_id),
                "class_id": int(active_tracks[internal_id]["cls_id"]),
                "confidence": float(confidence),
                "bbox": list(map(float, bbox)),
                "center": list(map(float, [center_x, center_y])),
                
            })

        for internal_id, data in list(active_tracks.items()):
            lost = False
            if not (data["active"]):
                if (internal_id not in reusable_ids):
                    lost=True
            elif frame_count - data["frame_count"] > 10:
                lost = True
                
            if lost:
                print("Lost,", internal_id, ",at,", frame_index)
                active_tracks[internal_id]["active"] = False
                reusable_ids.append(internal_id)
                external_ids_to_remove = [k for k, v in track_id_map.items() if v == internal_id]
                for ext_id in external_ids_to_remove:
                    del track_id_map[ext_id]

        tracking_data.append(frame_tracking_data)
        # Record the count of active tracks for this frame
        current_active_count = sum(1 for track in active_tracks.values() if track["active"])
        active_track_counts.append((frame_index, current_active_count))


    with open(output_json, 'w') as f:
        json.dump(tracking_data, f, indent=4)

    # Plot the number of active tracks per frame using matplotlib
    if active_track_counts:
        frames, counts = zip(*active_track_counts)
        plt.figure(figsize=(10, 6))
        plt.plot(frames, counts, marker='o', linestyle='-')
        plt.xlabel('Frame Index')
        plt.ylabel('Number of Active Tracks')
        plt.title('Active Tracks per Frame')
        plt.grid(True)
        plt.show()


    print(f"Tracking data with ByteTrack saved to {output_json}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_json> <output_json>")
        sys.exit(1)

    input_json = os.path.join('json_output', sys.argv[1])
    output_json = os.path.join('json_output', sys.argv[2])

    perform_tracking_from_json(input_json, output_json)

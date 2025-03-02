#!/usr/bin/env python3
import json
import sys
from tqdm import tqdm  # Import tqdm for progress bar

# Constant for the maximum frame index to process.
MAX_FRAME = 9000  # Change this value as needed

def transform_data(input_data, max_frame=MAX_FRAME):
    """
    Transforms the input JSON into a frame-by-frame JSON.
    """
    frames = {}

    # Iterate over each object in the input.
    for track_id_str, obj in tqdm(input_data.items(), desc="Processing Objects", unit="obj"):
        try:
            track_id = int(track_id_str)
        except ValueError:
            continue

        for path_obj in obj.get("paths", []):
            last_seen_frame = path_obj.get("last_seen_frame")
            team_index = path_obj.get("team_index")
            path_coords = path_obj.get("path", [])

            if last_seen_frame is None or team_index is None or not path_coords:
                continue

            num_points = len(path_coords)
            start_frame = last_seen_frame - num_points + 1

            for i, coord in enumerate(path_coords):
                frame_number = start_frame + i
                if frame_number > max_frame:
                    break

                if len(coord) < 3:
                    continue
                x, y = coord[1], coord[2]

                obj_dict = {
                    "track_id": track_id,
                    "class_id": team_index,
                    "confidence": 1,
                    "center": [x, y]
                }

                frames.setdefault(frame_number, []).append(obj_dict)

    output = []
    for frame_index in tqdm(range(0, max_frame + 1), desc="Processing Frames", unit="frame"):
        output.append({
            "frame_index": frame_index,
            "objects": frames.get(frame_index, [])
        })

    return output

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} input.json output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as f:
        input_data = json.load(f)

    output_data = transform_data(input_data)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    main()
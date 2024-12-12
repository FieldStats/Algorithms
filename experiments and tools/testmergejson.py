import json
import os

# File paths
RIGHT_INTERSECTIONS_JSON = "right_intersections.json"
RIGHT_NON_INTERSECTIONS_JSON = "right_non_intersections.json"
OUTPUT_MERGED_JSON = "rightmerged.json"

def merge_json_files(intersections_file, non_intersections_file, output_file):
    """
    Merge two JSON files containing frame data into a single JSON file.
    :param intersections_file: Path to the intersections JSON file.
    :param non_intersections_file: Path to the non-intersections JSON file.
    :param output_file: Path to save the merged JSON file.
    """
    if not os.path.exists(intersections_file):
        print(f"Error: File '{intersections_file}' not found.")
        return
    if not os.path.exists(non_intersections_file):
        print(f"Error: File '{non_intersections_file}' not found.")
        return

    # Load data from the JSON files
    with open(intersections_file, "r") as f:
        intersections_data = json.load(f)

    with open(non_intersections_file, "r") as f:
        non_intersections_data = json.load(f)

    # Merge the two lists of frames
    merged_data = {}

    for frame in intersections_data:
        frame_index = frame["frame_index"]
        if frame_index not in merged_data:
            merged_data[frame_index] = {"frame_index": frame_index, "objects": []}
        merged_data[frame_index]["objects"].extend(frame["objects"])

    for frame in non_intersections_data:
        frame_index = frame["frame_index"]
        if frame_index not in merged_data:
            merged_data[frame_index] = {"frame_index": frame_index, "objects": []}
        merged_data[frame_index]["objects"].extend(frame["objects"])

    # Convert merged_data back to a list sorted by frame_index
    merged_list = sorted(merged_data.values(), key=lambda x: x["frame_index"])

    # Save the merged data
    with open(output_file, "w") as f:
        json.dump(merged_list, f, indent=2)

    print(f"Merged data saved to {output_file}")

def main():
    merge_json_files(RIGHT_INTERSECTIONS_JSON, RIGHT_NON_INTERSECTIONS_JSON, OUTPUT_MERGED_JSON)

if __name__ == "__main__":
    main()

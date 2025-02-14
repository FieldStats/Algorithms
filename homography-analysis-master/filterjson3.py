import json
import os

# Input JSON file
INPUT_JSON_FILE = "merged_output_with_transformed_center.json"
OUTPUT_JSON_FILE = "borderfiltered_merged_output_with_transformed_center.json"

# Video dimensions
VIDEO_WIDTH = 400
VIDEO_HEIGHT = 300
BORDER_THRESHOLD = 15

def is_near_border(center, width, height, threshold, source):
    """
    Check if the center is near the border, excluding specific edges based on the source.
    """
    x, y = center

    if source == "left":
        # Exclude right border comparison
        return (
            x <= threshold or
            y <= threshold or y >= height - threshold
        )
    elif source == "right":
        # Exclude left border comparison
        return (
            x >= width - threshold or
            y <= threshold or y >= height - threshold
        )
    return False

def filter_json_by_border(input_file, output_file, width, height, threshold):
    """
    Filter objects from the input JSON file if their transformed_center
    coordinates are near the border, considering their source.
    """
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    # Load JSON data
    with open(input_file, "r") as f:
        data = json.load(f)

    filtered_data = []

    for frame in data:
        frame_index = frame["frame_index"]
        objects = frame.get("objects", [])

        # Filter objects based on transformed_center and source
        filtered_objects = [
            obj for obj in objects
            if not is_near_border(obj.get("transformed_center", [0, 0]), width, height, threshold, obj.get("source", "unknown"))
        ]

        # Add frame to filtered data if it contains any objects
        if filtered_objects:
            filtered_data.append({"frame_index": frame_index, "objects": filtered_objects})

    # Save the filtered JSON data
    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Filtered JSON saved to '{output_file}'.")

if __name__ == "__main__":
    filter_json_by_border(INPUT_JSON_FILE, OUTPUT_JSON_FILE, VIDEO_WIDTH, VIDEO_HEIGHT, BORDER_THRESHOLD)

import json
import os

def convert_json(input_file, output_file, ids_to_convert):
    """
    Convert parts of the JSON file corresponding to the given IDs.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
        ids_to_convert (list): List of IDs to convert.
    """
    # Load input JSON
    with open(input_file, "r") as f:
        input_json = json.load(f)

    # Precompute overall min start time and max end time
    all_start_times = []
    all_end_times = []

    for obj_id, data in input_json.items():
        if int(obj_id) in ids_to_convert:
            for path_obj in data["paths"]:
                path = path_obj["path"]
                last_seen_frame = path_obj["last_seen_frame"]

                end_time = last_seen_frame
                start_time = end_time - len(path)

                all_start_times.append(start_time)
                all_end_times.append(end_time)

    min_start_time = min(all_start_times) if all_start_times else None
    max_end_time = max(all_end_times) if all_end_times else None

    if min_start_time is None or max_end_time is None:
        print("No valid data found to process.")
        return

    output = {"tracklets": {}}  # Dictionary to hold the new JSON structure
    unique_id = 1  # Start creating IDs from 1 to n

    for obj_id, data in input_json.items():
        # Only process if the ID is in the list of IDs to convert
        if int(obj_id) in ids_to_convert:
            created_objects = []  # List to store created objects for this primary ID

            for path_obj in data["paths"]:
                path = path_obj["path"]
                last_seen_frame = path_obj["last_seen_frame"]

                start_xy = path[0]  # First coordinate
                end_xy = path[-1]  # Last coordinate

                # Calculate start and end times
                end_time = last_seen_frame
                start_time = end_time - len(path)

                # Create the tracklet object
                tracklet = {
                    "id": f"T{unique_id}",
                    "start_time": start_time,
                    "end_time": end_time,
                    "x_start": start_xy[0],
                    "y_start": start_xy[1],
                    "x_end": end_xy[0],
                    "y_end": end_xy[1],
                    "next_wrong": None  # Temporary placeholder
                }

                # Add the tracklet to the dictionary
                output["tracklets"][tracklet["id"]] = tracklet
                created_objects.append(tracklet)
                unique_id += 1

            # Update next_wrong field
            for i in range(len(created_objects) - 1):
                created_objects[i]["next_wrong"] = created_objects[i + 1]["id"]

            # Adjust the first and last tracklet times
            if created_objects:
                created_objects[0]["start_time"] = min_start_time
                created_objects[-1]["end_time"] = max_end_time

    # Print overall min start time and max end time
    print(f"\nOverall Min Start Time: {min_start_time}")
    print(f"Overall Max End Time: {max_end_time}")

    # Save the adjusted output JSON
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAdjusted JSON saved to {output_file}")

# File paths
input_file = os.path.join(os.getcwd(), "debug.json")
output_file = os.path.join(os.getcwd(), "filtered_hungarian.json")

# IDs to convert
ids_to_convert = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# Perform conversion
convert_json(input_file, output_file, ids_to_convert)

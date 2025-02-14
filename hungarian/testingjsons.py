import json

# Load the JSON data from the debug.json file
with open('debug.json', 'r') as file:
    data = json.load(file)

def validate_paths(data):
    for top_id, details in data.items():
        paths = details.get("paths", [])
        for i in range(len(paths) - 1):
            current_path = paths[i]
            next_path = paths[i + 1]
            
            current_last_seen_frame = current_path["last_seen_frame"]
            next_path_length = len(next_path["path"])  # Correctly get the length of the next path's array
            next_last_seen_frame = next_path["last_seen_frame"]

            # Validation condition
            if current_last_seen_frame + next_path_length > next_last_seen_frame:
                print(f"Validation failed for ID {top_id}: "
                      f"Current last_seen_frame ({current_last_seen_frame}) "
                      f"+ next path_length ({next_path_length}) "
                      f"is >= next last_seen_frame ({next_last_seen_frame}).")
                return False
    print("Validation passed for all paths.")
    return True

# Run validation
validate_paths(data)

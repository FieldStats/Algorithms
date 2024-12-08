import json
import numpy as np
import cv2

# File paths
INPUT_LEFT_JSON = "left_intersections.json"
INPUT_RIGHT_JSON = "right_intersections.json"
HOMOGRAPHY_MATRIX_LEFT = "al2_homography_matrix.txt"
HOMOGRAPHY_MATRIX_RIGHT = "al1_homography_matrix.txt"
OUTPUT_LEFT_JSON = "filtered_left_intersections.json"
OUTPUT_RIGHT_JSON = "filtered_right_intersections.json"

OFFSET = 340  # Offset in width for right video objects
N = 25  # Distance threshold for transformed coordinates comparison


def transform_point(point, homography_matrix):
    """Transform a single point using a homography matrix."""
    point = np.array([[point]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    return transformed_point[0][0]


def count_objects_by_color(objects, color):
    """Count objects with a specific color."""
    return sum(1 for obj in objects if obj.get("color") == color)


def compare_and_filter_objects(left_json, right_json, homography_matrix_left, homography_matrix_right, offset, threshold):
    filtered_left = []
    filtered_right = []

    left_frames = {frame["frame_index"]: frame for frame in left_json}
    right_frames = {frame["frame_index"]: frame for frame in right_json}

    all_frame_indices = set(left_frames.keys()).union(right_frames.keys())

    total_left_purple = sum(1 for frame in left_json for obj in frame["objects"] if obj.get("color") == "purple")
    total_right_orange = sum(1 for frame in right_json for obj in frame["objects"] if obj.get("color") == "orange")
    matched_left_objects = set()

    total_right_objects = 0
    matched_right_objects = 0

    for frame_index in all_frame_indices:
        left_objects = left_frames.get(frame_index, {"objects": []})["objects"]
        right_objects = right_frames.get(frame_index, {"objects": []})["objects"]

        total_right_objects += len(right_objects)
        new_right_objects = []

        for right_obj in right_objects:
            # Calculate middle bottom of right bbox
            right_bbox = right_obj["bbox"]
            right_bottom = [(right_bbox[0] + right_bbox[2]) / 2, right_bbox[3]]

            # Transform the middle bottom point of the right bbox
            right_transformed = transform_point(right_bottom, homography_matrix_right)
            right_transformed_with_offset = [right_transformed[0] + offset, right_transformed[1]]

            closest_left_obj = None
            min_distance = threshold

            for left_obj in left_objects:
                # Calculate middle bottom of left bbox
                left_bbox = left_obj["bbox"]
                left_bottom = [(left_bbox[0] + left_bbox[2]) / 2, left_bbox[3]]

                # Transform the middle bottom point of the left bbox
                left_transformed = transform_point(left_bottom, homography_matrix_left)

                # Calculate distance between transformed points
                distance = np.linalg.norm(
                    np.array(right_transformed_with_offset) - np.array(left_transformed)
                )

                if distance < min_distance:
                    closest_left_obj = left_obj
                    min_distance = distance

            if closest_left_obj:
                # Mark left object as matched
                matched_left_objects.add(closest_left_obj["track_id"])

                # Add the right object with yellow color to the new right JSON
                right_obj["color"] = "yellow"
                new_right_objects.append(right_obj)
                matched_right_objects += 1
            else:
                # Add unmatched right object with orange color
                right_obj["color"] = "orange"
                new_right_objects.append(right_obj)

        # Add unmatched left objects to the new left JSON with purple color
        unmatched_left_objects = [
            obj for obj in left_objects if obj["track_id"] not in matched_left_objects
        ]
        for obj in unmatched_left_objects:
            obj["color"] = "purple"
        filtered_left.append({"frame_index": frame_index, "objects": unmatched_left_objects})

        # Append the new frame data to the right JSON
        if new_right_objects:
            filtered_right.append({"frame_index": frame_index, "objects": new_right_objects})

    unmatched_left_count = sum(len(frame["objects"]) for frame in filtered_left)
    unmatched_right_count = total_right_objects - matched_right_objects

    print(f"Total purple objects before algorithm: {total_left_purple}")
    print(f"Total unmatched left objects: {unmatched_left_count}")
    print(f"Total purple objects after algorithm: {sum(1 for frame in filtered_left for obj in frame['objects'] if obj.get('color') == 'purple')}")
    print(f"Total orange objects before algorithm: {total_right_orange}")
    print(f"Total unmatched right objects: {unmatched_right_count}")
    print(f"Total orange objects after algorithm: {sum(1 for frame in filtered_right for obj in frame['objects'] if obj.get('color') == 'orange')}")

    return filtered_left, filtered_right


def main():
    # Load JSON data
    with open(INPUT_LEFT_JSON, "r") as f:
        left_json = json.load(f)
    with open(INPUT_RIGHT_JSON, "r") as f:
        right_json = json.load(f)

    # Load homography matrices
    homography_matrix_left = np.loadtxt(HOMOGRAPHY_MATRIX_LEFT, delimiter=' ')
    homography_matrix_right = np.loadtxt(HOMOGRAPHY_MATRIX_RIGHT, delimiter=' ')

    # Compare and modify objects
    filtered_left, filtered_right = compare_and_filter_objects(
        left_json, right_json, homography_matrix_left, homography_matrix_right, OFFSET, N
    )

    # Save modified JSONs
    with open(OUTPUT_LEFT_JSON, "w") as f:
        json.dump(filtered_left, f, indent=2)
    with open(OUTPUT_RIGHT_JSON, "w") as f:
        json.dump(filtered_right, f, indent=2)

    print("Modified JSONs have been saved.")


if __name__ == "__main__":
    main()

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
N = 30  # Distance threshold for transformed coordinates comparison


def transform_point(point, homography_matrix):
    """Transform a single point using a homography matrix."""
    point = np.array([[point]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    return transformed_point[0][0]


def compare_and_filter_objects(left_json, right_json, homography_matrix_left, homography_matrix_right, offset, threshold):
    """
    Compare and modify objects based on transformed middle bottom coordinates of bounding boxes.
    """
    filtered_left = []
    filtered_right = []

    left_frames = {frame["frame_index"]: frame for frame in left_json}
    right_frames = {frame["frame_index"]: frame for frame in right_json}

    all_frame_indices = set(left_frames.keys()).union(right_frames.keys())

    total_right_objects = 0
    matched_right_objects = 0

    for frame_index in all_frame_indices:
        left_objects = left_frames.get(frame_index, {"objects": []})["objects"]
        right_objects = right_frames.get(frame_index, {"objects": []})["objects"]

        total_right_objects += len(right_objects)

        new_left_objects = []
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
                # Update colors and add to new objects
                if closest_left_obj.get("color") == "purple":
                    closest_left_obj["color"] = "pink"
                if right_obj.get("color") == "orange":
                    right_obj["color"] = "yellow"

                new_left_objects.append(closest_left_obj)
                new_right_objects.append(right_obj)
                matched_right_objects += 1

        # Add remaining right objects that are unmatched
        for right_obj in right_objects:
            if right_obj not in new_right_objects:
                new_right_objects.append(right_obj)

        # Add remaining left objects that are unmatched
        for left_obj in left_objects:
            if left_obj not in new_left_objects:
                new_left_objects.append(left_obj)

        # Append the new frame data to filtered JSONs
        if new_left_objects:
            filtered_left.append({"frame_index": frame_index, "objects": new_left_objects})
        if new_right_objects:
            filtered_right.append({"frame_index": frame_index, "objects": new_right_objects})

    # Calculate and print matching percentage
    matching_percentage = (matched_right_objects / total_right_objects) * 100 if total_right_objects > 0 else 0
    print(f"{matching_percentage:.2f}% of right objects matched with a left object.")

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

import json
import numpy as np
import cv2

# File paths
JSON_LEFT_INTERSECTION = "left_intersections.json"
JSON_RIGHT_INTERSECTION = "right_intersections.json"
HOMOGRAPHY_MATRIX_LEFT = "al2_homography_matrix.txt"
HOMOGRAPHY_MATRIX_RIGHT = "al1_homography_matrix.txt"
DIMENSIONS_FILE = "dimensions.txt"
NEW_JSON_LEFT_INTERSECTION = "new_left_intersections.json"
NEW_JSON_RIGHT_INTERSECTION = "new_right_intersections.json"

def transform_point(point, homography_matrix):
    """Transform a single point using a homography matrix."""
    point = np.array([[point]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    return transformed_point[0][0]

def inverse_transform_point(point, homography_matrix):
    """Inverse transform a single point using a homography matrix."""
    inverse_matrix = np.linalg.inv(homography_matrix)
    return transform_point(point, inverse_matrix)

def adjust_center_coordinates(center, blue_line_x_src, blue_line_x_dst, homography_matrix_src, homography_matrix_dst, direction):
    """
    Adjust the x-coordinate of the center when copying to the opposite JSON file.
    - Calculate the x-coordinate difference between the center and the source blue line.
    - Add this difference to the destination blue line coordinate.
    - Use inverse transformation to calculate the final coordinates.
    """
    # Transform to world coordinates
    transformed_point = transform_point(center, homography_matrix_src)
    x_trans_src, y_trans_src = transformed_point

    # Calculate new transformed coordinates
    width_diff = x_trans_src - blue_line_x_src
    new_x_trans_dst = blue_line_x_dst + width_diff

    # Transform back to pixel coordinates
    inverse_transformed_point = inverse_transform_point((new_x_trans_dst, y_trans_src), homography_matrix_dst)
    new_x, new_y = map(float, inverse_transformed_point)  # Convert to Python float

    # Print debugging information
    print(f"[{direction}] Moving center:")
    print(f"  Original center (pixel): {center}")
    print(f"  Transformed coordinates (before): ({x_trans_src}, {y_trans_src})")
    print(f"  Transformed coordinates (after): ({new_x_trans_dst}, {y_trans_src})")
    print(f"  Adjusted center (pixel): ({new_x}, {new_y})")

    # Update center coordinates
    return [new_x, new_y]


def create_new_jsons(blue_line_left, blue_line_right, homography_matrix_left, homography_matrix_right):
    """Create new intersection JSONs without retaining previous data."""
    # Load original JSONs
    with open(JSON_LEFT_INTERSECTION, "r") as f:
        left_intersection = json.load(f)
    with open(JSON_RIGHT_INTERSECTION, "r") as f:
        right_intersection = json.load(f)

    # Initialize new JSONs
    new_left = []
    new_right = []

    # Helper function to copy crossing objects
    def copy_crossing_objects(frame_data, homography_matrix_src, homography_matrix_dst, blue_line_src, blue_line_dst, destination_json, is_left_side):
        """Copy objects crossing the blue line to the opposite JSON with adjusted center coordinates."""
        for obj in frame_data["objects"]:
            center = obj["center"]
            transformed_point = transform_point(center, homography_matrix_src)
            x_trans, _ = transformed_point

            # Determine if the object crosses the blue line
            if is_left_side and x_trans > blue_line_src:  # Crossed to the right
                new_center = adjust_center_coordinates(
                    center, blue_line_src, blue_line_dst, homography_matrix_src, homography_matrix_dst, "Left to Right"
                )
                new_obj = obj.copy()
                new_obj["center"] = new_center
                destination_json.append({
                    "frame_index": frame_data["frame_index"],
                    "objects": [new_obj]
                })
            elif not is_left_side and x_trans < blue_line_src:  # Crossed to the left
                new_center = adjust_center_coordinates(
                    center, blue_line_src, blue_line_dst, homography_matrix_src, homography_matrix_dst, "Right to Left"
                )
                new_obj = obj.copy()
                new_obj["center"] = new_center
                destination_json.append({
                    "frame_index": frame_data["frame_index"],
                    "objects": [new_obj]
                })

    # Process left intersection frames
    for frame_data in left_intersection:
        copy_crossing_objects(
            frame_data, homography_matrix_left, homography_matrix_right, blue_line_left, blue_line_right, new_right, is_left_side=True
        )

    # Process right intersection frames
    for frame_data in right_intersection:
        copy_crossing_objects(
            frame_data, homography_matrix_right, homography_matrix_left, blue_line_right, blue_line_left, new_left, is_left_side=False
        )

    # Save new JSONs
    with open(NEW_JSON_LEFT_INTERSECTION, "w") as f:
        json.dump(new_left, f, indent=2)
    with open(NEW_JSON_RIGHT_INTERSECTION, "w") as f:
        json.dump(new_right, f, indent=2)

    print("New JSONs have been saved.")

def main():
    # Load dimensions and homography matrices
    with open(DIMENSIONS_FILE, "r") as f:
        lines = f.readlines()
    blue_line_right, _ = map(int, lines[0].split())
    blue_line_left, _ = map(int, lines[1].split())

    homography_matrix_left = np.loadtxt(HOMOGRAPHY_MATRIX_LEFT, delimiter=' ')
    homography_matrix_right = np.loadtxt(HOMOGRAPHY_MATRIX_RIGHT, delimiter=' ')

    # Create new JSONs
    create_new_jsons(blue_line_left, blue_line_right, homography_matrix_left, homography_matrix_right)

if __name__ == "__main__":
    main()

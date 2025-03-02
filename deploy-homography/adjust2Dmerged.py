import json
import numpy as np
import cv2
import shutil
from tqdm import tqdm  # Import tqdm for progress tracking

# File paths
JSON_LEFT_INTERSECTION = "filtered_left_intersections.json"
JSON_RIGHT_INTERSECTION = "filtered_right_intersections.json"
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

def adjust_center_coordinates(bbox, center, blue_line_x_src, blue_line_x_dst, homography_matrix_src, homography_matrix_dst):
    """
    Adjust the coordinates of the new center based on the middle bottom of the bbox.
    """
    # Calculate middle bottom of the bbox
    middle_bottom = [(bbox[0] + bbox[2]) / 2, bbox[3]]  # (mid_x, bottom_y)

    # Compute difference vector between original center and middle bottom
    difference_vector = [center[0] - middle_bottom[0], center[1] - middle_bottom[1]]

    # Transform the middle bottom to world coordinates
    transformed_point = transform_point(middle_bottom, homography_matrix_src)
    x_trans_src, y_trans_src = transformed_point

    # Calculate new transformed coordinates
    width_diff = x_trans_src - blue_line_x_src
    new_x_trans_dst = blue_line_x_dst + width_diff

    # Transform back to pixel coordinates
    inverse_transformed_point = inverse_transform_point((new_x_trans_dst, y_trans_src), homography_matrix_dst)
    new_middle_bottom_x, new_middle_bottom_y = map(float, inverse_transformed_point)

    # Calculate the new center by adding the difference vector
    new_center_x = new_middle_bottom_x + difference_vector[0]
    new_center_y = new_middle_bottom_y + difference_vector[1]

    return [new_center_x, new_center_y]

def create_new_jsons(blue_line_left, blue_line_right, homography_matrix_left, homography_matrix_right):
    """Create new intersection JSONs by copying previous JSONs and inserting updated objects."""
    # Load original JSONs
    with open(JSON_LEFT_INTERSECTION, "r") as f:
        left_intersection = json.load(f)
    with open(JSON_RIGHT_INTERSECTION, "r") as f:
        right_intersection = json.load(f)

    # Copy original JSONs as base for new JSONs
    shutil.copy(JSON_LEFT_INTERSECTION, NEW_JSON_LEFT_INTERSECTION)
    shutil.copy(JSON_RIGHT_INTERSECTION, NEW_JSON_RIGHT_INTERSECTION)

    # Reload new JSONs after copying
    with open(NEW_JSON_LEFT_INTERSECTION, "r") as f:
        new_left = json.load(f)
    with open(NEW_JSON_RIGHT_INTERSECTION, "r") as f:
        new_right = json.load(f)

    # Helper function to copy and insert updated objects
    def copy_crossing_objects(frame_data, homography_matrix_src, homography_matrix_dst, blue_line_src, blue_line_dst, destination_json, is_left_side):
        """Copy objects crossing the blue line to the opposite JSON with adjusted center coordinates."""
        for obj in tqdm(frame_data["objects"], desc="Copying Objects", leave=False, unit="object"):
            bbox = obj["bbox"]
            center = obj["center"]

            # Check crossing based on middle bottom
            middle_bottom = [(bbox[0] + bbox[2]) / 2, bbox[3]]  # (mid_x, bottom_y)
            transformed_point = transform_point(middle_bottom, homography_matrix_src)
            x_trans, _ = transformed_point

            # Determine if the object crosses the blue line
            if (is_left_side and x_trans > blue_line_src) or (not is_left_side and x_trans < blue_line_src):
                new_center = adjust_center_coordinates(
                    bbox, center, blue_line_src, blue_line_dst, homography_matrix_src, homography_matrix_dst
                )
                new_obj = obj.copy()
                new_obj["center"] = new_center

                # Check if the frame exists in the destination JSON
                frame_entry = next((frame for frame in destination_json if frame["frame_index"] == frame_data["frame_index"]), None)
                if frame_entry:
                    # Add the object to the existing frame
                    frame_entry["objects"].append(new_obj)
                else:
                    # Create a new frame and add the object
                    new_frame = {
                        "frame_index": frame_data["frame_index"],
                        "objects": [new_obj]
                    }
                    destination_json.append(new_frame)

    # Process left intersection frames
    for frame_data in tqdm(left_intersection, desc="Processing Left Frames", unit="frame"):
        copy_crossing_objects(
            frame_data, homography_matrix_left, homography_matrix_right, blue_line_left, blue_line_right, new_right, is_left_side=True
        )

    # Process right intersection frames
    for frame_data in tqdm(right_intersection, desc="Processing Right Frames", unit="frame"):
        copy_crossing_objects(
            frame_data, homography_matrix_right, homography_matrix_left, blue_line_right, blue_line_left, new_left, is_left_side=False
        )

    # Save updated JSONs
    with open(NEW_JSON_LEFT_INTERSECTION, "w") as f:
        json.dump(new_left, f, indent=2)
    with open(NEW_JSON_RIGHT_INTERSECTION, "w") as f:
        json.dump(new_right, f, indent=2)

    print("Updated JSONs have been saved.")

def main():
    # Load dimensions and homography matrices
    with open(DIMENSIONS_FILE, "r") as f:
        lines = f.readlines()
    blue_line_right, _ = map(int, lines[0].split())
    blue_line_left, _ = map(int, lines[1].split())

    homography_matrix_left = np.loadtxt(HOMOGRAPHY_MATRIX_LEFT, delimiter=' ')
    homography_matrix_right = np.loadtxt(HOMOGRAPHY_MATRIX_RIGHT, delimiter=' ')

    # Create updated JSONs
    create_new_jsons(blue_line_left, blue_line_right, homography_matrix_left, homography_matrix_right)
    import bos
    print("third script completed. Now running the fourth script...")
    bos.main()  # Call the second script after finishing the first


import bos


if __name__ == "__main__":
    main()
    

    

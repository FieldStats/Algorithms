import json
import numpy as np
import cv2

# File paths
TRACKING_DATA_RIGHT = "tracking_data_1.json"
TRACKING_DATA_LEFT = "tracking_data_2.json"
HOMOGRAPHY_MATRIX_LEFT = "al2_homography_matrix.txt"
HOMOGRAPHY_MATRIX_RIGHT = "al1_homography_matrix.txt"
DIMENSIONS_FILE = "dimensions.txt"
OUTPUT_JSON = "possible_intersections.json"
DEBUG_IMAGE_RIGHT = "al1_debug.png"
DEBUG_IMAGE_LEFT = "al2_debug.png"

def load_dimensions_and_homographies():
    """Load blue line positions and homography matrices."""
    with open(DIMENSIONS_FILE, "r") as f:
        lines = f.readlines()
    blue_line_right, width_right = map(int, lines[0].split())
    blue_line_left, width_left = map(int, lines[1].split())

    homography_matrix_left = np.loadtxt(HOMOGRAPHY_MATRIX_LEFT, delimiter=' ')
    homography_matrix_right = np.loadtxt(HOMOGRAPHY_MATRIX_RIGHT, delimiter=' ')

    return blue_line_right, width_right, blue_line_left, width_left, homography_matrix_left, homography_matrix_right

def draw_red_lines(image_path, output_path, blue_line, red_line, inverse_homography_matrix):
    """
    Draw blue and red vertical lines on the untransformed (original) image for debugging.
    :param image_path: Path to the image file.
    :param output_path: Path to save the debug image.
    :param blue_line: Blue line x-coordinate in transformed space.
    :param red_line: Red line x-coordinate in transformed space.
    :param inverse_homography_matrix: Inverse homography matrix for untransforming coordinates.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Define the transformed lines as points in homogeneous coordinates
    height = image.shape[0]
    transformed_blue_line = np.array([[blue_line, 0, 1], [blue_line, height, 1]], dtype=np.float32).T
    transformed_red_line = np.array([[red_line, 0, 1], [red_line, height, 1]], dtype=np.float32).T

    # Untransform the points back to the original image space
    untransformed_blue_line = cv2.perspectiveTransform(transformed_blue_line.T[:, :2][np.newaxis, :, :], inverse_homography_matrix)[0]
    untransformed_red_line = cv2.perspectiveTransform(transformed_red_line.T[:, :2][np.newaxis, :, :], inverse_homography_matrix)[0]

    # Draw the blue line (untransformed coordinates)
    cv2.line(image, tuple(untransformed_blue_line[0].astype(int)), tuple(untransformed_blue_line[1].astype(int)), (255, 0, 0), 2)

    # Draw the red line (untransformed coordinates)
    cv2.line(image, tuple(untransformed_red_line[0].astype(int)), tuple(untransformed_red_line[1].astype(int)), (0, 0, 255), 2)

    # Save the debug image
    cv2.imwrite(output_path, image)
    print(f"Debug image saved to {output_path}")

def main():
    # Load dimensions, homographies, and offset
    blue_line_right, width_right, blue_line_left, width_left, homography_left, homography_right = load_dimensions_and_homographies()

    # Compute inverse homographies
    inverse_homography_left = np.linalg.inv(homography_left)
    inverse_homography_right = np.linalg.inv(homography_right)

    # Offset for red line calculation
    offset_x = blue_line_right  # First number from the first line of dimensions.txt

    # Compute red line indexes
    red_line_right = blue_line_right + offset_x
    red_line_left = blue_line_left - offset_x

    # Draw red lines in debug images
    draw_red_lines("al1.png", DEBUG_IMAGE_RIGHT, blue_line_right, red_line_right, inverse_homography_right)
    draw_red_lines("al2.png", DEBUG_IMAGE_LEFT, blue_line_left, red_line_left, inverse_homography_left)

    # Load tracking data
    with open(TRACKING_DATA_LEFT, "r") as f:
        data_left = json.load(f)
    with open(TRACKING_DATA_RIGHT, "r") as f:
        data_right = json.load(f)

    # Filter intersections
    intersections = []
    for frame_left, frame_right in zip(data_left, data_right):
        if frame_left["frame_index"] != frame_right["frame_index"]:
            continue

        frame_index = frame_left["frame_index"]
        filtered_objects = []

        # Check left video bounding boxes
        for obj in frame_left.get("objects", []):
            if obj["bbox"][2] > red_line_left and obj["bbox"][0] < width_left:  # Overlap with red line region
                filtered_objects.append(obj)

        # Check right video bounding boxes
        for obj in frame_right.get("objects", []):
            if obj["bbox"][0] < red_line_right and obj["bbox"][2] > 0:  # Overlap with red line region
                filtered_objects.append(obj)

        if filtered_objects:
            intersections.append({"frame_index": frame_index, "objects": filtered_objects})

    # Save the filtered intersections
    with open(OUTPUT_JSON, "w") as f:
        json.dump(intersections, f, indent=2)

    print(f"Filtered intersections saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

import cv2
import numpy as np

# List to store points
points = []
extra_point = None

# Colors for the lines
line_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Yellow

def select_points(event, x, y, flags, param):
    """
    Mouse callback function to capture points on mouse click.
    """
    global points, extra_point
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 8:
            points.append((x, y))
            print(f"Point selected: {x}, {y}")
            cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)  # Draw a small red circle for each point
            cv2.imshow("Select Points", temp_image)

            # Draw lines for every pair of points
            if len(points) % 2 == 0:
                pt1 = points[-2]
                pt2 = points[-1]
                line_index = (len(points) // 2) - 1
                color = line_colors[line_index % len(line_colors)]
                cv2.line(temp_image, pt1, pt2, color, 2)  # Draw line with the specified color
                cv2.imshow("Select Points", temp_image)
        elif len(points) == 8 and extra_point is None:
            extra_point = (x, y)
            print(f"Extra point selected: {x}, {y}")
            cv2.circle(temp_image, (x, y), 5, (255, 0, 0), -1)  # Draw a small blue circle for the extra point
            cv2.imshow("Select Points", temp_image)

def compute_intersection(p1, p2, p3, p4):
    """
    Compute the intersection of two lines (p1-p2 and p3-p4).
    """
    a1, b1, c1 = p2[1] - p1[1], p1[0] - p2[0], (p2[1] - p1[1]) * p1[0] + (p1[0] - p2[0]) * p1[1]
    a2, b2, c2 = p4[1] - p3[1], p3[0] - p4[0], (p4[1] - p3[1]) * p3[0] + (p3[0] - p4[0]) * p3[1]
    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        print("Lines are parallel!")
        return None  # Parallel lines

    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return int(x), int(y)

def save_homography_matrix(homography_matrix, filename):
    """
    Save the homography matrix to a text file.
    """
    np.savetxt(filename, homography_matrix, fmt='%.6f', delimiter=' ')
    print(f"Homography matrix saved to {filename}")

def process_image(image_path, output_image_path, output_matrix_path, output_dimensions_path):
    """
    Process an image to apply homography transformation and save the results.
    """
    global points, temp_image, extra_point
    points = []  # Reset points for the new image
    extra_point = None

    # Load the image
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"Error: Image {image_path} not found.")
        return

    temp_image = input_image.copy()
    cv2.imshow("Select Points", temp_image)
    cv2.setMouseCallback("Select Points", select_points)

    print(f"Select 8 points on the image '{image_path}' in pairs, followed by 1 extra blue point.")
    cv2.waitKey(0)

    if len(points) != 8 or extra_point is None:
        print("Error: You must select exactly 8 points and 1 extra point.")
        cv2.destroyAllWindows()
        return

    # Compute the intersections for specific line pairs
    intersections = []
    line_pairs = [
        (points[0], points[1], points[2], points[3]),
        (points[2], points[3], points[4], points[5]),
        (points[4], points[5], points[6], points[7]),
        (points[6], points[7], points[0], points[1])
    ]

    for p1, p2, p3, p4 in line_pairs:
        intersection = compute_intersection(p1, p2, p3, p4)
        if intersection:
            intersections.append(intersection)

    if len(intersections) < 4:
        print("Error: Could not find all 4 intersections.")
        cv2.destroyAllWindows()
        return

    # Perform homography transformation
    destination_points = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype=np.float32)
    source_points = np.array(intersections, dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(source_points, destination_points)

    # Save the homography matrix
    save_homography_matrix(homography_matrix, output_matrix_path)

    transformed_image = cv2.warpPerspective(input_image, homography_matrix, (400, 300))

    # Draw a vertical line from the transformed extra point
    transformed_extra_point = cv2.perspectiveTransform(np.array([[extra_point]], dtype=np.float32), homography_matrix)[0][0]
    transformed_extra_point = (int(transformed_extra_point[0]), int(transformed_extra_point[1]))
    cv2.line(transformed_image, (transformed_extra_point[0], 0), (transformed_extra_point[0], transformed_image.shape[0]), (255, 0, 0), 2)

    # Save the transformed image
    cv2.imwrite(output_image_path, transformed_image)
    print(f"Transformed image saved to {output_image_path}")

    # Save the width coordinate of the line and the image width
    with open(output_dimensions_path, "a") as f:
        f.write(f"{transformed_extra_point[0]} {transformed_image.shape[1]}\n")
    print(f"Line coordinate: {transformed_extra_point[0]}, Image width: {transformed_image.shape[1]}")
    cv2.destroyAllWindows()

def main():
    output_dimensions_path = "dimensions.txt"
    # Clear the output file
    with open(output_dimensions_path, "w") as f:
        f.write("")

    # Process the first image
    process_image("al1.png", "al1_transformed.png", "al1_homography_matrix.txt", output_dimensions_path)

    # Process the second image
    process_image("al2.png", "al2_transformed.png", "al2_homography_matrix.txt", output_dimensions_path)

if __name__ == "__main__":
    main()

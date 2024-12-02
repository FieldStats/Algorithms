import cv2
import numpy as np

# List to store points
points = []

# Colors for the lines
line_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Yellow

def select_points(event, x, y, flags, param):
    """
    Mouse callback function to capture points on mouse click.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
        points.append((x, y))
        print(f"Point selected: {x}, {y}")
        cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)  # Draw a small red circle for each point
        cv2.imshow("Select 8 Points", temp_image)

        # Draw lines for every pair of points
        if len(points) % 2 == 0:
            pt1 = points[-2]
            pt2 = points[-1]
            line_index = (len(points) // 2) - 1
            color = line_colors[line_index % len(line_colors)]
            cv2.line(temp_image, pt1, pt2, color, 2)  # Draw line with the specified color
            cv2.imshow("Select 8 Points", temp_image)

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

def draw_grid(transformed_image, homography_matrix, original_image):
    """
    Draw a grid on the transformed image and project it back onto the original image.
    """
    rows, cols = 300, 400  # Dimensions of the transformed image
    grid_lines = 20  # Number of divisions along each axis
    step_x = cols // grid_lines
    step_y = rows // grid_lines

    # Draw grid on the transformed image
    for i in range(0, cols + 1, step_x):  # Vertical lines
        cv2.line(transformed_image, (i, 0), (i, rows), (0, 0, 0), 1)

    for j in range(0, rows + 1, step_y):  # Horizontal lines
        cv2.line(transformed_image, (0, j), (cols, j), (0, 0, 0), 1)

    # Back-project the grid lines to the original image
    inverse_homography_matrix = np.linalg.inv(homography_matrix)
    for i in range(0, cols + 1, step_x):  # Vertical lines
        pt1 = cv2.perspectiveTransform(np.array([[[i, 0]]], dtype=np.float32), inverse_homography_matrix)
        pt2 = cv2.perspectiveTransform(np.array([[[i, rows]]], dtype=np.float32), inverse_homography_matrix)
        cv2.line(original_image, tuple(pt1[0][0].astype(int)), tuple(pt2[0][0].astype(int)), (0, 0, 0), 1)

    for j in range(0, rows + 1, step_y):  # Horizontal lines
        pt1 = cv2.perspectiveTransform(np.array([[[0, j]]], dtype=np.float32), inverse_homography_matrix)
        pt2 = cv2.perspectiveTransform(np.array([[[cols, j]]], dtype=np.float32), inverse_homography_matrix)
        cv2.line(original_image, tuple(pt1[0][0].astype(int)), tuple(pt2[0][0].astype(int)), (0, 0, 0), 1)

    cv2.imshow("Grid on Transformed Image", transformed_image)
    cv2.imshow("Grid Back-projected on Original Image", original_image)

def main():
    global points, temp_image
    
    # Load the image
    input_image = cv2.imread("al.jpg")
    if input_image is None:
        print("Error: Image not found.")
        return
    
    temp_image = input_image.copy()
    cv2.imshow("Select 8 Points", temp_image)
    cv2.setMouseCallback("Select 8 Points", select_points)
    
    print("Select 8 points on the image in pairs. Lines will connect each pair of points.")
    cv2.waitKey(0)

    if len(points) != 8:
        print("Error: You must select exactly 8 points.")
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
        return
    
    # Perform homography transformation
    destination_points = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype=np.float32)
    source_points = np.array(intersections, dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(source_points, destination_points)

    transformed_image = cv2.warpPerspective(input_image, homography_matrix, (400, 300))
    
    # Draw grid on the transformed and original images
    draw_grid(transformed_image, homography_matrix, input_image.copy())

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

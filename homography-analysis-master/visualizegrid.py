import cv2
import numpy as np

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

    return transformed_image, original_image

def main():
    # Load the homography matrix from the saved .txt file
    homography_matrix_file = "homography_matrix.txt"
    try:
        homography_matrix = np.loadtxt(homography_matrix_file, delimiter=' ')
        print(f"Homography matrix loaded from {homography_matrix_file}.")
    except Exception as e:
        print(f"Error loading homography matrix: {e}")
        return

    # Load the image
    input_image = cv2.imread("al2.png")
    if input_image is None:
        print("Error: Image not found.")
        return

    # Apply the homography transformation
    transformed_image = cv2.warpPerspective(input_image, homography_matrix, (400, 300))

    # Draw the grid on the transformed and original images
    grid_transformed_image, grid_original_image = draw_grid(
        transformed_image.copy(), homography_matrix, input_image.copy()
    )

    # Save the results
    cv2.imwrite("grid_transformed_image.jpg", grid_transformed_image)
    cv2.imwrite("grid_original_image.jpg", grid_original_image)
    print("Grid images saved: grid_transformed_image.jpg, grid_original_image.jpg")

    # Show the results
    cv2.imshow("Grid on Transformed Image", grid_transformed_image)
    cv2.imshow("Grid Back-projected on Original Image", grid_original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

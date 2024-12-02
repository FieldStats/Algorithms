import cv2
import numpy as np

# Global variables for images
original_image = None
grid_image = None
homography_matrix = None

def mark_point(event, x, y, flags, param):
    """
    Callback function to handle mouse clicks on the original image and mark the corresponding point on the grid image.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw the point on the original image
        cv2.circle(original_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Original Image", original_image)

        # Transform the point to the grid image
        point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(point, homography_matrix)
        tx, ty = transformed_point[0][0]

        # Draw the transformed point on the grid image
        cv2.circle(grid_image, (int(tx), int(ty)), 5, (0, 0, 255), -1)
        cv2.imshow("Grid Image", grid_image)

def main():
    global original_image, grid_image, homography_matrix

    # Load the original image
    original_image = cv2.imread("al.jpg")
    if original_image is None:
        print("Error: Original image not found.")
        return

    # Load the grid image
    grid_image = cv2.imread("grid_transformed_image.jpg")
    if grid_image is None:
        print("Error: Grid image not found.")
        return

    # Load the homography matrix from the saved .txt file
    homography_matrix_file = "homography_matrix.txt"
    try:
        homography_matrix = np.loadtxt(homography_matrix_file, delimiter=' ')
        print(f"Homography matrix loaded from {homography_matrix_file}.")
    except Exception as e:
        print(f"Error loading homography matrix: {e}")
        return

    # Create windows for displaying the images
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Grid Image", grid_image)

    # Set mouse callback for the original image
    cv2.setMouseCallback("Original Image", mark_point)

    print("Click on the Original Image to mark points on the Grid Image.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

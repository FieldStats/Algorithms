import cv2
import numpy as np

def process_image_with_blur_and_threshold(input_image, output_image, blur_kernel=(5, 5), threshold_value=127):
    """
    Apply Gaussian blur and thresholding to an image.

    Parameters:
        input_image (str): Path to the input image file.
        output_image (str): Path to save the processed image.
        blur_kernel (tuple): Size of the kernel for Gaussian blurring.
        threshold_value (int): Threshold value for binary thresholding.

    Returns:
        None
    """
    # Load the image
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image file {input_image}")
        return

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)

    # Apply binary thresholding
    _, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Save the processed image
    cv2.imwrite(output_image, thresholded)

    print(f"Processed image saved as {output_image}")

# Define input and output paths
input_image_path = "frame_0190.png"
output_image_path = "processed_image2.jpg"

# Apply Gaussian blur and thresholding
process_image_with_blur_and_threshold(
    input_image=input_image_path,
    output_image=output_image_path,
    blur_kernel=(5, 5),
    threshold_value=10
)

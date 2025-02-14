import cv2
import numpy as np
import json

def draw_best_circle(input_image, bbox_json_file, output_image):
    """
    Draw bounding boxes from a JSON file on the image and highlight the best circle-like box in orange.

    Parameters:
        input_image (str): Path to the original input image.
        bbox_json_file (str): Path to the JSON file containing bounding box data.
        output_image (str): Path to save the output image with the best circle highlighted.

    Returns:
        None
    """
    # Load the original image
    image = cv2.imread(input_image)

    if image is None:
        print(f"Error: Unable to load image file {input_image}")
        return

    # Load bounding box data from JSON
    with open(bbox_json_file, "r") as f:
        bbox_data = json.load(f)

    # Prepare to find the best circularity and aspect ratio
    best_score = -1
    best_bbox = None

    for bbox in bbox_data:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        # Ensure dimensions are valid
        if w == 0 or h == 0:
            continue

        # Calculate circularity: actual area vs. ideal circular area
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        actual_area = cv2.countNonZero(gray[y:y+h, x:x+w])
        radius = min(w, h) / 2.0
        full_circle_area = np.pi * (radius ** 2)
        circularity = (full_circle_area - abs(full_circle_area - actual_area)) / full_circle_area if full_circle_area > 0 else 0

        # Calculate aspect ratio score (favor aspect ratios close to 1, ideal for circles)
        aspect_ratio = w / h if h > 0 else 0
        aspect_ratio_score = 1 - abs(aspect_ratio - 1)

        # Combine circularity and aspect ratio into a single score
        combined_score = circularity * aspect_ratio_score

        # Draw the bounding box in green
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Update the best score and bounding box if necessary
        if combined_score > best_score:
            best_score = combined_score
            best_bbox = (x, y, w, h)

    # Highlight the best bounding box in orange
    if best_bbox:
        x, y, w, h = best_bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)  # Orange

    # Save the result image
    cv2.imwrite(output_image, image)

    print(f"Output image with best circle highlighted saved as {output_image}")

# Define input paths
input_image_path = "output_with_contours.jpg"
bbox_json_file_path = "adjusted_bounding_boxes.json"
output_image_path = "best_circle_highlighted.jpg"

# Draw bounding boxes and highlight the best circle-like one
draw_best_circle(
    input_image=input_image_path,
    bbox_json_file=bbox_json_file_path,
    output_image=output_image_path
)

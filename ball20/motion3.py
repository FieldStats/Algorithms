import cv2
import numpy as np
import json

def draw_contours_with_hull_offset(input_image, output_image, save_json_file, min_area=100, hull_offset=5):
    """
    Apply convex hull algorithm with offset to an image, merge intersecting hulls, draw contours in green, merged hulls in red,
    and save cropped bounding boxes for k-means clustering.

    Parameters:
        input_image (str): Path to the input image file.
        output_image (str): Path to save the output image with contours and hulls.
        save_json_file (str): Path to save adjusted bounding boxes as a JSON file.
        min_area (int): Minimum area of contours to consider as objects.
        hull_offset (int): Offset to expand or shrink the convex hull.

    Returns:
        None
    """
    # Load the binary image
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image file {input_image}")
        return

    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to color image for visualization
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # List to store merged hulls
    merged_hulls = []

    # List to store bounding box data for saving
    bbox_data = []

    # Iterate through contours
    for contour in contours:
        # Filter small contours by area
        if cv2.contourArea(contour) < min_area:
            continue

        # Draw the original contour in green
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 1)

        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] == 0:  # Avoid division by zero
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Apply offset to the contour points
        offset_contour_points = []
        for point in contour:
            x, y = point[0]
            # Offset the point outward (or inward if negative offset)
            dx = x - cx
            dy = y - cy
            x_new = int(x + hull_offset * dx / (np.sqrt(dx**2 + dy**2) + 1e-5))
            y_new = int(y + hull_offset * dy / (np.sqrt(dx**2 + dy**2) + 1e-5))
            offset_contour_points.append([[x_new, y_new]])

        offset_contour_points = np.array(offset_contour_points, dtype=np.int32)

        # Recalculate the hull for the offset contour points
        recalculated_hull = cv2.convexHull(offset_contour_points)

        # Check for intersections with existing merged hulls
        intersects = False
        for i, existing_hull in enumerate(merged_hulls):
            union_hull = np.concatenate((existing_hull, recalculated_hull))
            merged_hull = cv2.convexHull(union_hull)

            # If the merged hull's area is significantly larger than the sum of the two, they intersect
            existing_area = cv2.contourArea(existing_hull)
            new_area = cv2.contourArea(recalculated_hull)
            merged_area = cv2.contourArea(merged_hull)

            if merged_area <= existing_area + new_area + 1e-5:  # Small tolerance for floating-point errors
                merged_hulls[i] = merged_hull
                intersects = True
                break

        if not intersects:
            merged_hulls.append(recalculated_hull)

    # Draw the merged hulls in red and calculate bounding boxes
    for hull in merged_hulls:
        cv2.drawContours(output, [hull], -1, (0, 0, 255), 2)

        # Calculate bounding box for the hull
        x, y, w, h = cv2.boundingRect(hull)

        # Adjust the bounding box based on the hull offset
        x_new = x + hull_offset
        y_new = y + hull_offset
        w_new = max(1, w - 2 * hull_offset)  # Ensure width is positive
        h_new = max(1, h - 2 * hull_offset)  # Ensure height is positive

        # Save adjusted bounding box coordinates for later use
        bbox_data.append({"x": x_new, "y": y_new, "w": w_new, "h": h_new})

        # Draw the adjusted bounding box on the output image
        cv2.rectangle(output, (x_new, y_new), (x_new + w_new, y_new + h_new), (255, 255, 0), 2)  # Yellow

    # Save the bounding box data to a JSON file
    with open(save_json_file, "w") as f:
        json.dump(bbox_data, f, indent=4)

    print(f"Bounding boxes saved to {save_json_file}")

    # Save the result image
    cv2.imwrite(output_image, output)

    print(f"Output image with contours and merged hulls saved as {output_image}")

# Define input and output paths
input_image_path = "processed_image2.jpg"
output_image_path = "output_with_contours.jpg"
save_json_file_path = "adjusted_bounding_boxes.json"

# Draw contours with hull offset and save bounding boxes to JSON
draw_contours_with_hull_offset(
    input_image=input_image_path,
    output_image=output_image_path,
    save_json_file=save_json_file_path,
    min_area=30,
    hull_offset=10
)

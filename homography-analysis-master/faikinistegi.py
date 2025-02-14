import json
import cv2
import numpy as np

def draw_objects_on_videos(json_path, left_video_path, right_video_path, left_output_path, right_output_path):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Open video files
    left_cap = cv2.VideoCapture(left_video_path)
    right_cap = cv2.VideoCapture(right_video_path)

    if not left_cap.isOpened():
        print(f"Error: Could not open video file {left_video_path}")
        return

    if not right_cap.isOpened():
        print(f"Error: Could not open video file {right_video_path}")
        return

    # Get video properties (assuming both videos have the same properties)
    frame_width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(left_cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    left_out = cv2.VideoWriter(left_output_path, fourcc, fps, (frame_width, frame_height))
    right_out = cv2.VideoWriter(right_output_path, fourcc, fps, (frame_width, frame_height))

    frame_index = 0

    while left_cap.isOpened() and right_cap.isOpened():
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()

        if not left_ret or not right_ret:
            break

        # Filter objects for the current frame
        objects = [obj for obj in data if obj['frame_index'] == frame_index and obj['objects']]

        for obj_group in objects:
            for obj in obj_group['objects']:
                bbox = obj['bbox']
                color = obj['color']
                class_id = obj['class_id']

                # Convert color name to BGR
                bgr_color = {
                    "purple": (128, 0, 128),
                    "blue": (255, 0, 0),
                    "red": (0, 0, 255),
                    "yellow": (0, 255, 255),
                    "orange": (0, 165, 255)
                }.get(color.lower(), (0, 255, 0))

                # Draw on the appropriate frame
                if color.lower() in ["blue", "purple"]:
                    start_point = (int(bbox[0]), int(bbox[1]))
                    end_point = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(left_frame, start_point, end_point, bgr_color, 2)

                    # Add class ID and confidence as text
                    label = f"ID: {class_id}"
                    cv2.putText(left_frame, label, (start_point[0], start_point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)

                elif color.lower() in ["red", "yellow", "orange"]:
                    start_point = (int(bbox[0]), int(bbox[1]))
                    end_point = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(right_frame, start_point, end_point, bgr_color, 2)

                    # Add class ID and confidence as text
                    label = f"ID: {class_id}"
                    cv2.putText(right_frame, label, (start_point[0], start_point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)

        left_out.write(left_frame)
        right_out.write(right_frame)
        frame_index += 1

    left_cap.release()
    right_cap.release()
    left_out.release()
    right_out.release()
    print(f"Videos with annotations saved to {left_output_path} and {right_output_path}")

# Example usage
json_path = "65.json"  # Path to the JSON file
left_video_path = "video_leftlongshifted.mp4"  # Path to the left video
right_video_path = "video_rightlong.mp4"  # Path to the right video
left_output_path = "left_faik.mp4"  # Path to save the left output video
right_output_path = "right_faik.mp4"  # Path to save the right output video

draw_objects_on_videos(json_path, left_video_path, right_video_path, left_output_path, right_output_path)

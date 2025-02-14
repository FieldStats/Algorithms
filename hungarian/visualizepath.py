import json
import os
import cv2

def main():
    # 1. Load debug.json from current directory
    json_path = os.path.join(os.getcwd(), "debug.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Could not find debug.json in {os.getcwd()}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. Choose a top-level ID (string)
    top_level_id = "3"  # Adjust as needed
    if top_level_id not in data:
        raise KeyError(f"Key '{top_level_id}' not found in debug.json.")

    obj_data = data[top_level_id]
    paths = obj_data.get("paths", [])
    if not paths:
        print(f"No 'paths' found for top-level ID '{top_level_id}'.")
        return

    # 3. Open a video file (instead of an image)
    video_name = "transformed_merged_output.mp4"  # <-- Adjust as needed
    video_path = os.path.join(os.getcwd(), video_name)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Could not find video '{video_name}' in {os.getcwd()}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video '{video_name}'")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video writer
    output_name = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    # Minimum number of points to consider a path
    MIN_POINTS = 3

    # Define color cycle for each PATH: [RED, YELLOW, GREEN] in BGR
    color_cycle = [
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (0, 255, 0)     # Green
    ]

    # 4. Process the video frame-by-frame
    while True:
        ret, frame = cap.read()
        if not ret:
            # No more frames or failed to read
            break

        # Convert the frame to grayscale and back to 3 channels
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bw_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Get frame dimensions
        img_height, img_width = bw_frame.shape[:2]

        # For each path, draw lines on this frame
        for index, path_info in enumerate(paths):
            coordinates = path_info['path']

            # Skip if path is too short
            if len(coordinates) < MIN_POINTS:
                continue

            # Pick a color for this entire path
            path_color = color_cycle[index % len(color_cycle)]

            # Draw line segments
            for i in range(len(coordinates) - 1):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[i + 1]

                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)

                # Check if points are inside the frame
                if (0 <= x1 < img_width and 0 <= y1 < img_height and
                    0 <= x2 < img_width and 0 <= y2 < img_height):
                    # Draw a thin line (thickness=1)
                    cv2.line(bw_frame, (x1, y1), (x2, y2), path_color, thickness=1)

        # Write the modified frame to the output video
        out.write(bw_frame)

    # Cleanup
    cap.release()
    out.release()

    print(f"Video saved as '{output_name}'")

if __name__ == "__main__":
    main()

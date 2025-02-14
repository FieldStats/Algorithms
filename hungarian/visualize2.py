import json
import os
import cv2
import numpy as np

def build_lines_for_video(paths, color_cycle):
    """
    Given:
      - paths: list of dicts, each with:
          {
            'path': [ [x0,y0], [x1,y1], ... ],
            'last_seen_frame': <int>
          }
      - color_cycle: list of BGR tuples, e.g. [(0,0,255), (0,255,0), ...]

    Returns:
      lines_for_frame: dict mapping frame_index -> list of (color, (x1,y1), (x2,y2))

    Frame logic:
      - Suppose a path has K coordinates.
      - The last coordinate is at 'last_seen_frame'.
      - The second-to-last coordinate is at (last_seen_frame - 1), etc.
      - So coordinate i is at frame = last_seen_frame - ((K-1) - i).
      - We store the line from (i-1) to i at that frame index.
    """
    lines_for_frame = {}

    for p_idx, path_info in enumerate(paths):
        coords = path_info.get("path", [])
        last_frame = path_info.get("last_seen_frame", 0)
        K = len(coords)
        if K < 2:
            continue  # Can't draw lines with fewer than 2 points

        # Choose a color for this path from the cycle
        color = color_cycle[p_idx % len(color_cycle)]

        for i in range(1, K):
            x1, y1 = coords[i - 1]
            x2, y2 = coords[i]
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            # Frame index for coordinate i
            frame_i = last_frame - ((K - 1) - i)

            if frame_i not in lines_for_frame:
                lines_for_frame[frame_i] = []
            lines_for_frame[frame_i].append((color, (x1, y1), (x2, y2)))

    return lines_for_frame

def overlay_lines(base_frame, lines_image):
    """
    Overlays all non-black pixels from lines_image onto base_frame.
    Both images should be the same size and 3-channel.
    Returns a new image with lines drawn on the base_frame.
    """
    result = base_frame.copy()
    # Convert lines_image to grayscale to find non-black regions
    gray_mask = cv2.cvtColor(lines_image, cv2.COLOR_BGR2GRAY)
    # Where mask > 0, we copy the color from lines_image
    mask_indices = np.where(gray_mask > 0)
    result[mask_indices] = lines_image[mask_indices]
    return result

def main():
    # ----------------------------------------------------------------------
    # 1. Load debug.json
    # ----------------------------------------------------------------------
    json_path = os.path.join(os.getcwd(), "debug.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Could not find debug.json in {os.getcwd()}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # ----------------------------------------------------------------------
    # 2. IDs from 1..23 (as strings) & define a color cycle
    # ----------------------------------------------------------------------
    # We can use a short cycle of 3 colors. Adjust or add more if you want:
    base_color_cycle = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 255, 0)   # Cyan / Yellow in BGR
    ]
    
    id_list = [str(i) for i in range(1, 24)]  # IDs "1" .. "23"

    # Build lines_for_video by merging all IDs
    lines_for_video = {}

    for track_id in id_list:
        if track_id not in data:
            continue
        paths = data[track_id].get("paths", [])
        lines_dict = build_lines_for_video(paths, base_color_cycle)
        # Merge lines_dict into lines_for_video
        for frame_i, line_list in lines_dict.items():
            if frame_i not in lines_for_video:
                lines_for_video[frame_i] = []
            lines_for_video[frame_i].extend(line_list)

    if not lines_for_video:
        print("No lines found for IDs 1..23. Exiting.")
        return

    # Gather min/max frames from the JSON data
    all_frames = sorted(lines_for_video.keys())
    min_frame = all_frames[0]
    max_frame = all_frames[-1]
    print(f"Lines cover frames from {min_frame} to {max_frame}")

    # ----------------------------------------------------------------------
    # 3. Read an input video
    # ----------------------------------------------------------------------
    video_input = "transformed_merged_output.mp4"  # <--- Adjust to your actual video file
    if not os.path.isfile(video_input):
        raise FileNotFoundError(f"Could not find video '{video_input}' in {os.getcwd()}")

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise IOError(f"Failed to open video '{video_input}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # default to 30 if zero

    # ----------------------------------------------------------------------
    # 4. Create output video writer (MP4)
    # ----------------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (vid_width, vid_height))

    print(f"Reading '{video_input}' with {total_frames} frames at {fps:.2f} fps.")
    print("Lines once drawn will accumulate in subsequent frames.")

    # ----------------------------------------------------------------------
    # 5. Accumulate lines over frames
    # ----------------------------------------------------------------------
    # We'll keep a single lines_image that starts black (0) and draw new lines
    # onto it for each frame. This ensures lines persist once they've appeared.
    lines_image = np.zeros((vid_height, vid_width, 3), dtype=np.uint8)

    current_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames

        # Convert to grayscale, back to BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_for_drawing = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # If we have lines for this frame index, draw them onto lines_image
        if current_frame_idx in lines_for_video:
            for (color, (x1, y1), (x2, y2)) in lines_for_video[current_frame_idx]:
                if (0 <= x1 < vid_width and 0 <= y1 < vid_height and
                    0 <= x2 < vid_width and 0 <= y2 < vid_height):
                    cv2.line(lines_image, (x1, y1), (x2, y2), color, 2)

        # Overlay the accumulated lines onto the current grayscale frame
        final_frame = overlay_lines(frame_for_drawing, lines_image)
        out.write(final_frame)

        current_frame_idx += 1
        # If we've processed all frames in the video, we stop
        if current_frame_idx >= total_frames:
            break

    cap.release()
    out.release()
    print("Done! Video saved as 'output.mp4'.")

if __name__ == "__main__":
    main()

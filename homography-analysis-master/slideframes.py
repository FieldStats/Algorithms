import json
import cv2
import os

# Configuration
n = 25  # Number of frames to shift/remove
input_json = "tracking_data_2.json"  # Path to the JSON file
input_video = "video_leftlong.mp4"  # Path to the input video
output_json = "tracking_data_2shifted.json"  # Path for the output JSON
output_video = "video_leftlongshifted.mp4"  # Path for the output video

# Process JSON file
with open(input_json, "r") as json_file:
    data = json.load(json_file)

# Remove frames less than n and reindex the remaining frames
shifted_data = []
for frame in data:
    if frame["frame_index"] >= n:
        frame["frame_index"] -= n
        shifted_data.append(frame)

# Save the updated JSON file
with open(output_json, "w") as json_file:
    json.dump(shifted_data, json_file, indent=4)

print(f"Shifted JSON saved to {output_json}")

# Process Video
cap = cv2.VideoCapture(input_video)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the frame range to keep
start_frame = n
end_frame = total_frames

if start_frame >= total_frames:
    print(f"Error: The shift value ({n}) exceeds the total number of frames ({total_frames}).")
    cap.release()
else:
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Skip the first `n` frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Write the remaining frames to the output video
    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    # Confirm video is saved
    if os.path.exists(output_video):
        print(f"Trimmed video saved to {output_video}")
    else:
        print("Error: Trimmed video could not be saved.")

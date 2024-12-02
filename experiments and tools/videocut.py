import cv2

# Input video file
input_video = "video_rightlong.mp4"
output_video = "output_1sec.mp4"

# Open the input video
video_capture = cv2.VideoCapture(input_video)

if not video_capture.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# Get the frame rate (fps) of the original video
fps = video_capture.get(cv2.CAP_PROP_FPS)

# Video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
output_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Extract and write the first 60 frames
frame_count = 0

while frame_count < 60:
    success, frame = video_capture.read()
    if not success:
        print("Reached the end of the video or encountered an issue.")
        break
    output_writer.write(frame)
    frame_count += 1

# Release resources
video_capture.release()
output_writer.release()

print(f"Saved the first 60 frames as a 1-second video: {output_video}")

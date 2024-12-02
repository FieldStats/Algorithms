import os
import cv2
import sys

def extract_frames(videoname):
    """
    Extract frames from a video file and save them as PNG images in the current directory.

    :param videoname: Name of the video file (e.g., "new_test2.mp4").
    """
    # Construct video path
    video_path = os.path.join('../../track_saver/input_videos', videoname)
    output_dir = '.'

    # Verify video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        sys.exit(1)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}.")
        sys.exit(1)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Generate filename with zero-padded frame index
        frame_filename = os.path.join(output_dir, f"frame_{frame_index:04d}.png")

        # Save frame as a PNG file
        cv2.imwrite(frame_filename, frame)
        frame_index += 1

        print(f"Saved: {frame_filename}")

    cap.release()
    print(f"All frames extracted and saved.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <videoname>")
        sys.exit(1)

    videoname = sys.argv[1]
    extract_frames(videoname)

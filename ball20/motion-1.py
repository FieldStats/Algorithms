import cv2
from tqdm import tqdm

def crop_video_bottom(input_video, output_video):
    """
    Crop the input video to the bottom half and save the cropped video.

    Parameters:
        input_video (str): Path to the input video file.
        output_video (str): Path to save the cropped video.

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height // 2))

    # Process each frame with a progress bar
    with tqdm(total=total_frames, desc="Cropping Video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the frame to the bottom half
            cropped_frame = frame[frame_height // 2:, :]

            # Write the cropped frame to the output video
            out.write(cropped_frame)

            # Update progress bar
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()
    print(f"Cropped video saved as {output_video}")

# Define input and output paths
input_video_path = "video_leftlongshifted.mp4"
output_video_path = "video_leftlongshiftedcrop.mp4"

# Crop the video
crop_video_bottom(input_video_path, output_video_path)

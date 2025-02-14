import cv2
from tqdm import tqdm  # For progress bar

def process_video_with_colors(input_video, output_video):
    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    # Use MP4 codec (H.264) for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Initialize CPU-based background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    print("Processing video...")

    # Initialize progress bar
    with tqdm(total=total_frames, desc="Progress", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply background subtraction
            fg_mask = fgbg.apply(frame)

            # Convert mask to color
            mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

            # Blend the mask with the original frame
            highlighted_frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

            # Write the processed frame to the output video
            out.write(highlighted_frame)

            # Update the progress bar
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()
    print(f"Processing complete. Saved to: {output_video}")

# Input and output file paths
input_video = "input_video.mp4"  # Replace with your input video path
output_video = "processed_video.mp4"  # Desired output file path (MP4)

# Run the processing function
process_video_with_colors(input_video, output_video)

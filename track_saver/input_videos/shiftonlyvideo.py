import cv2
from tqdm import tqdm

def crop_first_n_frames(input_video_path, output_video_path, n):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Skip the first n frames
    for _ in tqdm(range(n), desc="Skipping frames"):
        ret = cap.grab()
        if not ret:
            print("Reached end of video while skipping frames.")
            break

    # Write the remaining frames to the output video
    frame_count = 0
    with tqdm(total=total_frames - n, desc="Writing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()

    print(f"Cropped {n} frames from the beginning of {input_video_path}.")
    print(f"Saved the result as {output_video_path} with {frame_count} frames.")

if __name__ == "__main__":
    input_video = "left5.mp4"
    output_video = "left5shifted.mp4"
    frames_to_crop = 59  # Change this value to the number of frames you want to crop

    crop_first_n_frames(input_video, output_video, frames_to_crop)

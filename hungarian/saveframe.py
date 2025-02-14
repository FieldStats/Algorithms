import cv2

def save_500th_frame(videopath):
    """
    Reads a video from the given path and saves the 500th frame as 'frame500.jpg' in the current directory.
    """
    cap = cv2.VideoCapture(videopath)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {videopath}")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:  # If the video ends or cannot read the frame
            print("Error: Reached end of video or cannot read the frame.")
            break
        
        frame_count += 1
        
        if frame_count == 500:
            output_image_name = "frame500.jpg"
            cv2.imwrite(output_image_name, frame)
            print(f"Saved the 500th frame as {output_image_name}")
            break
    
    cap.release()

# Define your video path here
videopath = "transformed_merged_output.mp4"  # Replace with your video file path
save_500th_frame(videopath)

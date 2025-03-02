import cv2
import numpy as np

def fade_green_colors(frame, green_saturation_factor=0.5, non_green_brightness_factor=1.2):
    """
    Process a frame by desaturating greenish colors while enhancing non-green brightness.
    - green_saturation_factor: Multiplier for saturation of greenish colors.
    - non_green_brightness_factor: Multiplier for brightness of non-green colors.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Extract H, S, V channels
    h, s, v = cv2.split(hsv)

    # Define greenish hue range (approx. 30–90 in OpenCV’s 0-179 scale)
    green_mask = (h > 30) & (h < 90)

    # Reduce saturation for greenish colors
    s[green_mask] *= green_saturation_factor

    # Boost brightness for non-green areas
    v[~green_mask] *= non_green_brightness_factor

    # Clip values to valid range
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)

    # Merge back and convert to BGR
    hsv_processed = cv2.merge([h, s, v])
    frame_processed = cv2.cvtColor(hsv_processed.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return frame_processed

def play_video_with_filter(video_path, green_saturation_factor=0.5, non_green_brightness_factor=1.2):
    """
    Play video in real-time with a toggleable filter.
    Press 'f' to toggle the filter ON/OFF.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    filter_enabled = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply the filter only if enabled
        if filter_enabled:
            frame = fade_green_colors(frame, green_saturation_factor, non_green_brightness_factor)

        # Display the frame
        cv2.imshow("Video Player (Press 'f' to Toggle Filter, 'q' to Quit)", frame)

        # Capture key press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('f'):  # Toggle filter ON/OFF
            filter_enabled = not filter_enabled
            print("Filter ON" if filter_enabled else "Filter OFF")
        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    input_file = "transformed_merged_output.mp4"
    play_video_with_filter(input_file)

if __name__ == "__main__":
    main()

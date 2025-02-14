import cv2

# Replace this with your video file path
video_file = 'left5shifted.mp4'

# Open the video file
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read one frame from the video
ret, frame = cap.read()
if not ret:
    print("Error: Could not read a frame from the video.")
    cap.release()
    exit()

# Get frame dimensions
# frame.shape returns a tuple (height, width, channels)
height, width, channels = frame.shape

# Count the total number of pixels
total_pixels = width * height

print(f"Frame dimensions: {width} x {height} (Width x Height)")
print(f"Total number of pixels: {total_pixels}")

# Optionally, show the frame (press any key to close)
cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video capture object
cap.release()

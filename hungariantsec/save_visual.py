import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from matplotlib.widgets import Button, Slider
import matplotlib as mpl
import numpy as np


def visualize_tracking_data_and_save_video(json_path, video_path=None, confidence_threshold=0.1, output_video_path="output.mp4"):
    """
    Visualize tracking data from a JSON file, overlay it on video frames, and save the combined video.

    :param json_path: Path to the JSON file with tracking data.
    :param video_path: Path to the video file (optional, enables overlay if provided).
    :param confidence_threshold: Minimum confidence score to display objects (default 0.1).
    :param output_video_path: Path to save the output video.
    """
    # Load tracking data from JSON
    with open(json_path, 'r') as json_file:
        tracking_data = json.load(json_file)

    # Known classes and assigned distinct colors
    known_classes = [-1, 0, 1, 2, 3]  # Replace with actual class IDs
    class_colors = {cls: plt.cm.tab10(i / len(known_classes)) for i, cls in enumerate(known_classes)}

    # Initialize video capture if video path is provided
    cap = cv2.VideoCapture(video_path) if video_path else None
    
    if not cap or not cap.isOpened():
        print("Failed to open video file.")
        return

    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Set DPI to match the video resolution
    dpi = 100
    fig_width = frame_width / dpi
    fig_height = frame_height / dpi

    # Create figure with exact size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all margins
    ax.set_facecolor('black')

    for frame_idx in range(min(total_frames, len(tracking_data))):
        print("Processing... ", frame_idx)
        ax.clear()  # Clear previous plot
        ax.set_facecolor('black')

        frame_info = tracking_data[frame_idx]

        # Read video frame for overlay
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read video frame {frame_idx}.")
            break

        # Plot objects in the current frame
        for obj in frame_info["objects"]:
            class_id = obj["class_id"]
            center = obj["center"]

            track_id = obj.get("track_id", "N/A")  # Get track ID or default to "N/A"

            if obj["confidence"] < confidence_threshold:
                continue

            # Plot center of the bounding box with a white contour
            color = class_colors.get(class_id, 'white')  # Default to white if class is unknown
            ax.scatter(
                center[0], center[1],
                color=color,         # Fill color
                edgecolors='white',  # White contour
                linewidth=1.5,       # Thickness of the contour
                s=100,               # Size of the marker
            )

            # Annotate track ID near the detection point
            ax.text(
                (center[0]) - 7, center[1] - 5,  # Offset text position slightly
                f"ID: {track_id}",
                color='white',
                fontsize=8,
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1)
            )

        # Add a transparent overlay of the video frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb, alpha=0.9, extent=[0, frame.shape[1], frame.shape[0], 0])

        # Render the plot to a numpy array
        fig.canvas.draw()
        plot_frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        plot_frame = plot_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Write combined frame to output video
        out.write(cv2.cvtColor(plot_frame, cv2.COLOR_RGB2BGR))

    # Release resources
    cap.release()
    out.release()
    plt.close(fig)
    print(f"Output video saved to {output_video_path}")



if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(f"Usage: python {sys.argv[0]} <jsonname> <videoname> [confidence_threshold] [output_video_path]")
        sys.exit(1)

    jsonname = sys.argv[1]
    videoname = sys.argv[2]
    confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    output_video_path = sys.argv[4] if len(sys.argv) > 4 else "output.mp4"

    json_path = os.path.join('./', jsonname)
    video_path = os.path.join('./', videoname)

    visualize_tracking_data_and_save_video(json_path, video_path, confidence_threshold, output_video_path)

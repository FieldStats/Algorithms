import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import json

# ---------------------------
# Load positions and mapping data.
# ---------------------------
with open("positions.json", "r") as f:
    positions_data = json.load(f)
positions_data.sort(key=lambda entry: entry["frame"])

with open("output2.json", "r") as f:
    mapping_list = json.load(f)
mapping_list.sort(key=lambda event: event["frame"])

# ---------------------------
# Video capture initialization.
# ---------------------------
video_path = "bos5.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file.")
    exit(1)

# ---------------------------
# Mapping function.
# ---------------------------
def get_mapping_for_frame(current_frame):
    current_mapping = {}
    for event in mapping_list:
        if event["frame"] <= current_frame:
            current_mapping = event["mapping"]
        else:
            break
    return current_mapping

# ---------------------------
# Formation Positions Loader
# ---------------------------
def load_formation_positions(file_path):
    """
    Loads formation positions from a text file.
    
    Expected file format:
      Left Team Formation: <formation_type>
      Player 1: (x, y)
      Player 2: (x, y)
      ...
      
      Right Team Formation: <formation_type>
      Player 1: (x, y)
      Player 2: (x, y)
      ...
      
    Returns two dictionaries:
      left_positions: {player_index: (x, y), ...}
      right_positions: {player_index: (x, y), ...}
    """
    left_positions = {}
    right_positions = {}
    current_team = None
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Left Team Formation:"):
            current_team = "left"
            continue
        elif line.startswith("Right Team Formation:"):
            current_team = "right"
            continue
        elif line.startswith("Player"):
            try:
                # Expecting line like: "Player 1: (x, y)"
                parts = line.split(":")
                index_part = parts[0].split()  # ["Player", "1"]
                player_index = int(index_part[1])
                coord_str = parts[1].strip().strip("()")
                x_str, y_str = coord_str.split(",")
                x = float(x_str.strip())
                y = float(y_str.strip())
                if current_team == "left":
                    left_positions[player_index] = (x, y)
                elif current_team == "right":
                    right_positions[player_index] = (x, y)
            except Exception as e:
                print("Error parsing line:", line, e)
    return left_positions, right_positions

# Load formation positions from file (this file should already exist)
formation_file = "formation_positions.txt"
left_formation, right_formation = load_formation_positions(formation_file)

# ---------------------------
# Tkinter GUI setup.
# ---------------------------
root = tk.Tk()
root.title("Frame Viewer with Mapped IDs")

# Create a label to display the frame number
frame_label = tk.Label(root, text="Frame: 0", font=("Arial", 14))
frame_label.pack()

# Create an image label for displaying the frame
image_label = tk.Label(root)
image_label.pack()

current_index = 0  # Global index for positions_data

def show_frame():
    """Display the current frame with drawn player points, mapped IDs, and formation dots."""
    global current_index
    if current_index < 0 or current_index >= len(positions_data):
        print("Frame index out of range.")
        return

    frame_info = positions_data[current_index]
    frame_number = frame_info["frame"]
    player_positions = frame_info["positions"]
    mapping = get_mapping_for_frame(frame_number)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {frame_number}")
        return

    # Draw mapped player positions from positions_data.
    for player_id, pos in player_positions.items():
        x, y = pos
        center = (int(x), int(y))
        # Map the ID if present, otherwise use the original ID.
        mapped_id = int(mapping.get(str(player_id), player_id))
        
        # Determine the color based on ID range.
        if 1 <= mapped_id <= 11:
            color = (255, 255, 255)  # White
        elif 12 <= mapped_id <= 22:
            color = (0, 100, 0)      # Dark Green
        elif mapped_id == 23:
            color = (0, 255, 255)    # Yellow
        else:
            color = (0, 0, 255)      # Default Red for unexpected IDs

        # Draw a circle and the mapped ID text.
        cv2.circle(frame, center, 5, color, -1)
        cv2.putText(frame, str(mapped_id), (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # -------------------------------------------
    # Draw formation positions from the file.
    # Left team: dots in red with index.
    for idx, pos in left_formation.items():
        x, y = pos
        center = (int(x), int(y))
        # Draw left team dot (red)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # Write the index using a distinct font.
        cv2.putText(frame, str(idx + 1), (center[0] + 10, center[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Right team: dots in blue with index+12 and different font.
    for idx, pos in right_formation.items():
        x, y = pos
        center = (int(x), int(y))
        # Draw right team dot (blue)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
        # Write the index with an offset of 12, using a different font.
        display_index = idx + 12
        cv2.putText(frame, str(display_index), (center[0] + 10, center[1] + 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Convert frame to RGB and display in Tkinter.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    image_label.imgtk = imgtk  # Retain reference.
    image_label.config(image=imgtk)

    # Update frame label text.
    frame_label.config(text=f"Frame: {frame_number}")

def next_frame():
    """Advance to the next frame in positions_data."""
    global current_index
    if current_index < len(positions_data) - 1:
        current_index += 1
        show_frame()
    else:
        print("Reached the end of available frames.")

def prev_frame():
    """Return to the previous frame in positions_data."""
    global current_index
    if current_index > 0:
        current_index -= 1
        show_frame()
    else:
        print("Already at the first frame.")

# ---------------------------
# Create navigation buttons.
# ---------------------------
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

prev_button = ttk.Button(button_frame, text="Previous", command=prev_frame)
prev_button.pack(side="left", padx=5)

next_button = ttk.Button(button_frame, text="Next", command=next_frame)
next_button.pack(side="left", padx=5)

# Show the first frame on startup.
show_frame()

# Start the Tkinter event loop.
root.mainloop()

# Release the video capture when finished.
cap.release()

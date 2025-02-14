import cv2
import tkinter as tk
from tkinter import Label, Button, Toplevel
from PIL import Image, ImageTk

class VideoFrameViewer:
    def __init__(self, root, video_path):
        self.root = root
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_index = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # UI Elements
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.prev_button = Button(self.button_frame, text="Prev", command=self.show_prev_frame)
        self.prev_button.pack(side=tk.LEFT)

        self.prev_10_button = Button(self.button_frame, text="-10", command=self.show_prev_10_frames)
        self.prev_10_button.pack(side=tk.LEFT)

        self.prev_100_button = Button(self.button_frame, text="-100", command=self.show_prev_100_frames)
        self.prev_100_button.pack(side=tk.LEFT)

        self.frame_label = Label(self.button_frame, text=f"Frame: {self.frame_index}/{self.total_frames}")
        self.frame_label.pack(side=tk.LEFT)

        self.next_10_button = Button(self.button_frame, text="+10", command=self.show_next_10_frames)
        self.next_10_button.pack(side=tk.LEFT)

        self.next_100_button = Button(self.button_frame, text="+100", command=self.show_next_100_frames)
        self.next_100_button.pack(side=tk.LEFT)

        self.next_button = Button(self.button_frame, text="Next", command=self.show_next_frame)
        self.next_button.pack(side=tk.LEFT)

        self.label = Label(root)
        self.label.pack()

        self.show_frame(self.frame_index)

    def show_frame(self, index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))  # Resize frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            self.frame_label.config(text=f"Frame: {self.frame_index + 1}/{self.total_frames}")
        else:
            print(f"Failed to retrieve frame {index}")

    def show_prev_frame(self):
        if self.frame_index > 0:
            self.frame_index -= 1
            self.show_frame(self.frame_index)

    def show_next_frame(self):
        if self.frame_index < self.total_frames - 1:
            self.frame_index += 1
            self.show_frame(self.frame_index)

    def show_prev_10_frames(self):
        self.frame_index = max(0, self.frame_index - 10)
        self.show_frame(self.frame_index)

    def show_prev_100_frames(self):
        self.frame_index = max(0, self.frame_index - 100)
        self.show_frame(self.frame_index)

    def show_next_10_frames(self):
        self.frame_index = min(self.total_frames - 1, self.frame_index + 10)
        self.show_frame(self.frame_index)

    def show_next_100_frames(self):
        self.frame_index = min(self.total_frames - 1, self.frame_index + 100)
        self.show_frame(self.frame_index)

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Main Window")
    root.geometry("200x100")

    def open_left_video():
        left_window = Toplevel(root)
        left_window.title("Video Frame Viewer - left5.mp4")
        VideoFrameViewer(left_window, "left5shifted.mp4")

    def open_right_video():
        right_window = Toplevel(root)
        right_window.title("Video Frame Viewer - right5.mp4")
        VideoFrameViewer(right_window, "right5.mp4")

    Button(root, text="Open Left Video", command=open_left_video).pack(pady=10)
    Button(root, text="Open Right Video", command=open_right_video).pack(pady=10)

    root.mainloop()

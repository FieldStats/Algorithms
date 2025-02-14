import json
import os
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")  # or whichever backend you prefer

JSON_PATH = "tracklets.json"

class Tracklet:
    """
    Simple container for tracklet info.
    """
    def __init__(self, tid, start_time, end_time,
                 x_start, y_start, x_end, y_end,
                 next_wrong=None):
        self.id = tid
        # Timeline
        self.start_time = start_time
        self.end_time = end_time
        # XY
        self.x_start = x_start
        self.y_start = y_start
        self.x_end   = x_end
        self.y_end   = y_end
        # Pointer
        self.next_wrong = next_wrong

    def duration(self):
        return self.end_time - self.start_time

    def __repr__(self):
        return (f"Tracklet({self.id}, "
                f"t=[{self.start_time}, {self.end_time}], "
                f"next_wrong={self.next_wrong})")


def load_tracklets(json_path=JSON_PATH):
    """
    Load tracklets from JSON if it exists, otherwise create sample tracklets.
    """
    if not os.path.exists(json_path):
        print(f"No {json_path} found; creating default tracklets.")
        return {
            "T1": Tracklet("T1", 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, next_wrong="T2"),
            "T2": Tracklet("T2", 1.0, 2.5, 1.0, 1.0, 2.0, 2.0, next_wrong=None),
            "T3": Tracklet("T3", 0.5, 1.5, 2.0, 1.0, 3.0, 2.0, next_wrong="T1"),
        }

    with open(json_path, "r") as f:
        data = json.load(f)
    tracklets = {}
    for tid, info in data["tracklets"].items():
        t = Tracklet(
            tid=tid,
            start_time=info["start_time"],
            end_time=info["end_time"],
            x_start=info["x_start"],
            y_start=info["y_start"],
            x_end=info["x_end"],
            y_end=info["y_end"],
            next_wrong=info["next_wrong"]
        )
        tracklets[tid] = t

    print(f"Loaded {len(tracklets)} tracklets from {json_path}")
    return tracklets


def save_tracklets(tracklets, json_path=JSON_PATH):
    """
    Save tracklets to JSON so that changes persist across runs.
    """
    data = {"tracklets": {}}
    for tid, t in tracklets.items():
        data["tracklets"][tid] = {
            "id": t.id,
            "start_time": t.start_time,
            "end_time": t.end_time,
            "x_start": t.x_start,
            "y_start": t.y_start,
            "x_end": t.x_end,
            "y_end": t.y_end,
            "next_wrong": t.next_wrong
        }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(tracklets)} tracklets to {json_path}")


# ============ Setup global state ===========

tracklets = load_tracklets()  # from JSON or default

fig, (ax_timeline, ax_xy) = plt.subplots(1, 2, figsize=(12, 6))

# We'll keep references for timeline bars
timeline_bars = {}
MODE_MOVE_ALL = "move_all"
MODE_MOVE_LEFT = "move_left"
MODE_MOVE_RIGHT = "move_right"

dragging_rect = None
dragging_tracklet = None
dragging_mode = None
initial_click_x = None

# For XY dragging
xy_lines = {}
xy_pts = {}
dragging_xy_point = None
dragging_xy_trk = None
dragging_xy_idx = None  # 0 => start, 1 => end

# We'll store the last two tracklets clicked in timeline:
# [ older, newer ]
selected_for_connection = []


# --------------------------------
# 1) Redraw Function
# --------------------------------

def redraw_all():
    # ========== TIMELINE ==========
    ax_timeline.clear()
    ax_timeline.set_title("Timeline: drag bar center to shift, edges to extend")
    ax_timeline.set_xlabel("Time")
    ax_timeline.set_ylabel("Row (each tracklet)")

    for i, (tid, trk) in enumerate(tracklets.items()):
        left = trk.start_time
        width = trk.duration()

        bar = ax_timeline.barh(y=i, width=width, left=left,
                               height=0.4, color="skyblue")[0]
        timeline_bars[tid] = bar

        # Label
        ax_timeline.text(left + width * 0.5, i, tid,
                         ha='center', va='center', color='black')

        # Arrow if next_wrong is set
        if trk.next_wrong:
            nxt = tracklets.get(trk.next_wrong)
            if nxt:
                j = list(tracklets.keys()).index(nxt.id)
                ax_timeline.annotate(
                    "",
                    xy=(nxt.start_time, j),
                    xytext=(trk.end_time, i),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2)
                )

    ax_timeline.set_ylim(-1, len(tracklets) + 0.5)
    ax_timeline.set_xlim(-0.5, 5)
    ax_timeline.grid(True)

    # ========== XY SPACE ==========
    ax_xy.clear()
    ax_xy.set_title("XY Space (drag endpoints). IDs labeled.")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.grid(True)

    for tid, trk in tracklets.items():
        line, = ax_xy.plot(
            [trk.x_start, trk.x_end],
            [trk.y_start, trk.y_end],
            color="blue", marker='o'
        )
        xy_lines[tid] = line

        p1 = ax_xy.plot(trk.x_start, trk.y_start, 'ro', picker=5)[0]
        p2 = ax_xy.plot(trk.x_end,   trk.y_end,   'go', picker=5)[0]
        xy_pts[tid] = (p1, p2)

        # label tracklet ID near midpoint
        mx = 0.5 * (trk.x_start + trk.x_end)
        my = 0.5 * (trk.y_start + trk.y_end)
        ax_xy.text(mx, my + 0.1, tid, color='darkblue',
                   ha='center', va='bottom', fontsize=9)

    ax_xy.set_xlim(-0.5, 5)
    ax_xy.set_ylim(-0.5, 5)

    fig.canvas.draw_idle()

# --------------------------------
# 2) Timeline Helpers
# --------------------------------

def pick_timeline_mode(trk, click_x):
    threshold = 0.1
    dist_left = abs(click_x - trk.start_time)
    dist_right = abs(click_x - trk.end_time)
    if dist_left < threshold:
        return MODE_MOVE_LEFT
    elif dist_right < threshold:
        return MODE_MOVE_RIGHT
    else:
        return MODE_MOVE_ALL


def timeline_drag(trk, mode, new_x):
    global initial_click_x
    if initial_click_x is None:
        return
    dx = new_x - initial_click_x
    initial_click_x = new_x

    if mode == MODE_MOVE_ALL:
        trk.start_time += dx
        trk.end_time += dx
    elif mode == MODE_MOVE_LEFT:
        trk.start_time += dx
    elif mode == MODE_MOVE_RIGHT:
        trk.end_time += dx


# --------------------------------
# 3) Mouse / Key Handlers
# --------------------------------

def on_click(event):
    global dragging_rect, dragging_tracklet, dragging_mode
    global initial_click_x
    global dragging_xy_point, dragging_xy_trk, dragging_xy_idx

    if event.inaxes == ax_timeline:
        if event.xdata is None:
            return
        # Check bars
        for tid, trk in tracklets.items():
            bar = timeline_bars[tid]
            contains, _ = bar.contains(event)
            if contains:
                dragging_rect = bar
                dragging_tracklet = trk
                initial_click_x = event.xdata
                dragging_mode = pick_timeline_mode(trk, event.xdata)

                # Track the last 2 distinct clicked in [older, newer]
                if len(selected_for_connection) == 2:
                    # If the new click is already in the list, reorder or do nothing
                    if trk in selected_for_connection:
                        # Move it to the end
                        selected_for_connection.remove(trk)
                        selected_for_connection.append(trk)
                    else:
                        # Remove the old first, then add new
                        selected_for_connection.pop(0)
                        selected_for_connection.append(trk)
                else:
                    # If we have <2 items
                    if trk not in selected_for_connection:
                        selected_for_connection.append(trk)

                break

    elif event.inaxes == ax_xy:
        if event.xdata is None or event.ydata is None:
            return
        # Check endpoints
        for tid, trk in tracklets.items():
            p1, p2 = xy_pts[tid]
            c1, _ = p1.contains(event)
            c2, _ = p2.contains(event)
            if c1:
                dragging_xy_point = p1
                dragging_xy_trk = trk
                dragging_xy_idx = 0
                break
            if c2:
                dragging_xy_point = p2
                dragging_xy_trk = trk
                dragging_xy_idx = 1
                break


def on_release(event):
    global dragging_rect, dragging_tracklet, dragging_mode
    global initial_click_x
    global dragging_xy_point, dragging_xy_trk, dragging_xy_idx

    dragging_rect = None
    dragging_tracklet = None
    dragging_mode = None
    initial_click_x = None

    dragging_xy_point = None
    dragging_xy_trk = None
    dragging_xy_idx = None


def on_motion(event):
    # Timeline
    global dragging_rect, dragging_tracklet, dragging_mode
    if dragging_rect is not None and dragging_tracklet is not None:
        if event.xdata is not None:
            timeline_drag(dragging_tracklet, dragging_mode, event.xdata)
            redraw_all()
        return

    # XY
    global dragging_xy_point, dragging_xy_trk, dragging_xy_idx
    if dragging_xy_point is not None and dragging_xy_trk is not None:
        if event.xdata is not None and event.ydata is not None:
            if dragging_xy_idx == 0:
                dragging_xy_trk.x_start = event.xdata
                dragging_xy_trk.y_start = event.ydata
            else:
                dragging_xy_trk.x_end = event.xdata
                dragging_xy_trk.y_end = event.ydata
            redraw_all()
        return


def on_key(event):
    """
    Key commands:
      b => break between (older, newer)
      c => connect older->newer
      e => save to JSON
    """
    if event.key == 'b':
        if len(selected_for_connection) == 2:
            older, newer = selected_for_connection  # older is [0], newer is [1]
            # Break if older->newer or newer->older
            if older.next_wrong == newer.id:
                older.next_wrong = None
                print(f"Removed next_wrong from {older.id} -> {newer.id}.")
            elif newer.next_wrong == older.id:
                newer.next_wrong = None
                print(f"Removed next_wrong from {newer.id} -> {older.id}.")
            else:
                print("No direct pointer found to break.")
            redraw_all()
        else:
            print("Need 2 tracklets (clicked in timeline) to break.")

    elif event.key == 'c':
        if len(selected_for_connection) == 2:
            older, newer = selected_for_connection  # older is [0], newer is [1]
            # Connect older->newer
            older.next_wrong = newer.id
            print(f"Connected {older.id} -> {newer.id}")
            redraw_all()
        else:
            print("Need 2 tracklets (clicked in timeline) to connect.")

    elif event.key == 'e':
        print("Saving tracklets to JSON (key=e pressed)...")
        save_tracklets(tracklets, JSON_PATH)


# --------------------------------
# Connect events & run
# --------------------------------

fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("key_press_event", on_key)

redraw_all()
plt.show()

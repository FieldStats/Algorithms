import json
import itertools
import matplotlib.pyplot as plt

def calculate_iou(bbox1, bbox2):
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection

    return intersection / union if union != 0 else 0

def find_max_iou_per_frame(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    max_ious = {}

    for frame in data:
        frame_index = frame['frame_index']
        objects = frame['objects']
        bboxes = [obj['bbox'] for obj in objects]

        max_iou = 0
        for bbox1, bbox2 in itertools.combinations(bboxes, 2):
            iou = calculate_iou(bbox1, bbox2)
            max_iou = max(max_iou, iou)

        max_ious[frame_index] = max_iou

    return max_ious

if __name__ == "__main__":
    input_file = "team_borderfiltered_merged_output_with_transformed_center.json"  # Replace with the path to your JSON file
    max_ious = find_max_iou_per_frame(input_file)

    for frame, max_iou in max_ious.items():
        print(f"Frame {frame}: Max IoU = {max_iou:.4f}")

    # Calculate and print max IoU per 100 frames
    max_ious_per_100_frames = {}
    for i in range(0, max(max_ious.keys()) + 1, 100):
        max_iou_in_batch = max(
            [iou for frame, iou in max_ious.items() if i <= frame < i + 100],
            default=0
        )
        max_ious_per_100_frames[f"Frames {i}-{i+99}"] = max_iou_in_batch

    print("\nMax IoU per 100 frames:")
    for batch, max_iou in max_ious_per_100_frames.items():
        print(f"{batch}: Max IoU = {max_iou:.4f}")

    # Plot number of IoUs higher than thresholds
    thresholds = [0.7, 0.8, 0.9, 0.95]
    counts = {threshold: 0 for threshold in thresholds}

    for iou in max_ious.values():
        for threshold in thresholds:
            if iou > threshold:
                counts[threshold] += 1

    plt.bar([str(threshold) for threshold in thresholds], counts.values())
    plt.title("Number of IoUs Above Thresholds")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Count")
    plt.show()

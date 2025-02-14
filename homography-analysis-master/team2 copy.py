import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def downscale_bbox_for_right_video(bbox):
    scale_factor = 1080 / 1080
    return [int(coord * scale_factor) for coord in bbox]

def get_player_team(image, team_assigner):
    top_half_image = image[0:int(image.shape[0] / 2), :]
    downsampled_image = top_half_image

    image_2d = downsampled_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300)
    kmeans.fit(image_2d)

    labels = kmeans.labels_
    clustered_image = labels.reshape(downsampled_image.shape[0], downsampled_image.shape[1])
    corner_clusters = [
        clustered_image[0, 0], clustered_image[0, -1],
        clustered_image[-1, 0], clustered_image[-1, -1]
    ]
    non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
    player_cluster = 1 - non_player_cluster

    player_color = kmeans.cluster_centers_[player_cluster]
    distances = kmeans.transform(player_color.reshape(1, -1))

    team_id = np.argmin(distances) + 1
    confidence = 1 - (distances[0][team_id - 1] / np.sum(distances))

    return team_id, round(confidence, 2)

def validate_source_and_color(obj):
    color = obj["color"].lower()
    source = obj["source"].lower()

    if color in ["yellow", "orange", "red"] and source != "right":
        print(f"Warning: Mismatch - Color {color} expected from 'right', found in {source}.")
    elif color in ["purple", "blue"] and source != "left":
        print(f"Warning: Mismatch - Color {color} expected from 'left', found in {source}.")

def display_images(json_data, left_video_path, right_video_path):
    left_cap = cv2.VideoCapture(left_video_path)
    right_cap = cv2.VideoCapture(right_video_path)

    left_count, right_count = 0, 0

    for frame_data in json_data:
        frame_index = frame_data["frame_index"]

        for obj in frame_data["objects"]:
            validate_source_and_color(obj)

            video_cap = left_cap if obj["source"] == "left" else right_cap

            if obj["source"] == "right":
                bbox = downscale_bbox_for_right_video(obj["bbox"])
            else:
                bbox = obj["bbox"]

            x1, y1, x2, y2 = map(int, bbox)

            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = video_cap.read()

            if not success:
                print(f"Failed to read frame {frame_index} from {obj['source']} video.")
                continue

            cropped_image = frame[y1:y2, x1:x2]
            team_id, confidence = get_player_team(cropped_image, None)

            # Display first 10 images for debugging
            if obj["source"] == "left" and left_count < 10:
                plt.figure()
                plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                plt.title(f"Left: Team {team_id}, Conf: {confidence}")
                left_count += 1

            if obj["source"] == "right" and right_count < 10:
                plt.figure()
                plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                plt.title(f"Right: Team {team_id}, Conf: {confidence}")
                right_count += 1

            if left_count >= 10 and right_count >= 10:
                break

        if left_count >= 10 and right_count >= 10:
            break

    left_cap.release()
    right_cap.release()
    plt.show()

if __name__ == "__main__":
    input_json_path = "borderfiltered_merged_output_with_transformed_center.json"
    left_video_path = "video_leftlongshifted.mp4"
    right_video_path = "video_rightlong.mp4"

    with open(input_json_path, "r") as f:
        json_data = json.load(f)

    display_images(json_data, left_video_path, right_video_path)

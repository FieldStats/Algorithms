import cv2
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, max_iter=300)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, image):
        top_half_image = image[0:int(image.shape[0] / 2), :]
        downsampled_image = cv2.resize(top_half_image, (top_half_image.shape[1] // 2, top_half_image.shape[0] // 2))
        kmeans = self.get_clustering_model(downsampled_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(downsampled_image.shape[0], downsampled_image.shape[1])
        corner_clusters = [
            clustered_image[0, 0], clustered_image[0, -1],
            clustered_image[-1, 0], clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, images):
        player_colors = [self.get_player_color(image) for image in images]
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300)
        kmeans.fit(np.array(player_colors))
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, image, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id], self.player_team_dict[f"{player_id}_confidence"]
        player_color = self.get_player_color(image)
        distances = self.kmeans.transform(player_color.reshape(1, -1))
        team_id = np.argmin(distances) + 1
        confidence = 1 - (distances[0][team_id - 1] / np.sum(distances))
        self.player_team_dict[player_id] = team_id
        self.player_team_dict[f"{player_id}_confidence"] = confidence
        return team_id, confidence

def downscale_bbox_for_right_video(bbox):
    scale_factor = 720 / 1080
    return [int(coord * scale_factor) for coord in bbox]

def process_frame(video_path, frame_index, objects, team_assigner, source):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Failed to read frame {frame_index} from {video_path}")

    for obj in objects:
        bbox = obj["bbox"]
        if source == "right":
            bbox = downscale_bbox_for_right_video(bbox)
        x1, y1, x2, y2 = map(int, bbox)
        cropped_image = frame[y1:y2, x1:x2]
        team_id, confidence = team_assigner.get_player_team(cropped_image, id(obj))
        obj["team_index"] = team_id
        obj["accuracy"] = round(confidence, 2)

def main(input_json_path, left_video_path, right_video_path, output_json_path):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    team_assigner = TeamAssigner()

    def process_frame_data(frame_data):
        video_path = left_video_path if frame_data["objects"][0]["color"] in ["purple", "blue"] else right_video_path
        source = frame_data["objects"][0]["source"]
        process_frame(video_path, frame_data["frame_index"], frame_data["objects"], team_assigner, source)

    with ThreadPoolExecutor() as executor:
        executor.map(process_frame_data, data)

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    input_json_path = "input.json"
    left_video_path = "left_video.mp4"
    right_video_path = "right_video.mp4"
    output_json_path = "output.json"

    main(input_json_path, left_video_path, right_video_path, output_json_path)

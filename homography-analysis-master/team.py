import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
import os

# Set the environment variable to limit the number of threads for MKL
os.environ["OMP_NUM_THREADS"] = "1"

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform CPU-based K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, max_iter=300)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, image):
        # Focus on the top half of the image
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Downsample the top half image for faster processing
        downsampled_image = cv2.resize(top_half_image, (top_half_image.shape[1] // 2, top_half_image.shape[0] // 2))

        # Get Clustering model
        kmeans = self.get_clustering_model(downsampled_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the downsampled image shape
        clustered_image = labels.reshape(downsampled_image.shape[0], downsampled_image.shape[1])

        # Simplified background removal using corner clusters
        corner_clusters = [
            clustered_image[0, 0], clustered_image[0, -1],
            clustered_image[-1, 0], clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, images):
        player_colors = []
        for image in images:
            player_color = self.get_player_color(image)
            player_colors.append(player_color)

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
        
        # Find the closest team cluster
        team_id = np.argmin(distances)
        team_id += 1  # Adjust team ID to start from 1

        # Calculate confidence as 1 - normalized distance
        confidence = 1 - (distances[0][team_id - 1] / np.sum(distances))

        # Store the results
        self.player_team_dict[player_id] = team_id
        self.player_team_dict[f"{player_id}_confidence"] = confidence

        return team_id, confidence

# New script to classify cropped images in batches and display results
def classify_images_in_folder(folder_path):
    team_assigner = TeamAssigner()
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    
    # Sort the images
    image_files = sorted(image_files)

    cropped_images = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        cropped_images.append(frame)

    # Initialize team colors using all images
    team_assigner.assign_team_color(cropped_images)

    # Display images in batches of 10
    batch_size = 10
    num_batches = len(cropped_images) // batch_size + (1 if len(cropped_images) % batch_size != 0 else 0)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(cropped_images))

        fig, axes = plt.subplots(1, end_idx - start_idx, figsize=(15, 5))
        if end_idx - start_idx == 1:
            axes = [axes]

        for i, idx in enumerate(range(start_idx, end_idx)):
            image = cropped_images[idx]
            team_id, confidence = team_assigner.get_player_team(image, player_id=idx + 1)

            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            axes[i].imshow(image_rgb)
            axes[i].set_title(f"Team {team_id}\nConf: {confidence:.2f}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    folder_path = "./cropped_objects_left"
    classify_images_in_folder(folder_path)


import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self, team_colors_file):
        self.team_colors = self.load_team_colors(team_colors_file)

    def load_team_colors(self, filename):
        """Load team colors from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Loaded team colors from {filename}: {data}")
        return {
            1: np.array(data["team_colors"]["team_1"]),
            2: np.array(data["team_colors"]["team_2"])
        }

    def get_clustering_model(self, image):
        """Perform KMeans clustering on an image."""
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def extract_player_color(self, image, debug=False):
        """Extract the dominant player color using KMeans and background subtraction."""
        # Crop top-half of the image
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Perform KMeans clustering
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Debugging: Show clustering labels as an image (only for debug=True)
        if debug:
            plt.figure()
            plt.imshow(labels, cmap="viridis")
            plt.title("Clustering Labels (Top Half)")
            plt.axis("off")
            plt.colorbar(label="Cluster")
            plt.show()

        # Identify the background cluster using corner pixels
        corner_clusters = [
            labels[0, 0],  # Top-left
            labels[0, -1],  # Top-right
            labels[-1, 0],  # Bottom-left
            labels[-1, -1],  # Bottom-right
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Get the dominant player color (cluster centroid)
        player_color = kmeans.cluster_centers_[player_cluster]
        if debug:
            print(f"Extracted Player Color (RGB): {player_color}")
        return player_color

    def classify_player(self, image, debug=False):
        """Classify the player based on the closest team color."""
        # Ensure correct color space (convert to RGB if needed)
        player_color = self.extract_player_color(image, debug=debug)

        # Compare the player's color with each team's color
        distances = {
            team_id: np.linalg.norm(player_color - color)
            for team_id, color in self.team_colors.items()
        }
        # Choose the closest team
        assigned_team = min(distances, key=distances.get)

        if debug:
            print(f"Distances to Teams: {distances}")
            print(f"Assigned Team: {assigned_team}")
        return assigned_team, player_color

    def debug_visualization(self, image, assigned_team, player_color):
        """Show debugging information for a single image."""
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Player Image")
        plt.axis("off")

        # Player color
        plt.subplot(1, 3, 2)
        plt.imshow([[player_color / 255]])
        plt.title("Extracted Player Color")
        plt.axis("off")

        # Assigned team color
        plt.subplot(1, 3, 3)
        plt.imshow([[self.team_colors[assigned_team] / 255]])
        plt.title(f"Assigned Team: {assigned_team}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# Initialize TeamAssigner with the saved team colors
team_assigner = TeamAssigner("team_colors.json")

# Path to the folder containing 100 cropped player images
image_folder = "cropped_objects_left"  # Replace with the actual folder path
image_files = sorted(os.listdir(image_folder))[:100]  # Get the first 100 images

# Loop through the images and classify them
classified_results = []

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # Enable debugging only for the first image
    debug = idx == 0

    # Classify the player
    assigned_team, player_color = team_assigner.classify_player(image, debug=debug)
    classified_results.append({
        "image": image_file,
        "team": assigned_team,
        "player_color": player_color.tolist()
    })

    # Show debugging visualization only for the first image
    if debug:
        print(f"Classifying {image_file}: Assigned to Team {assigned_team}")
        team_assigner.debug_visualization(image, assigned_team, player_color)

# Save the classification results to a JSON file
output_file = "classification_results.json"
with open(output_file, 'w') as f:
    json.dump(classified_results, f, indent=4)

print(f"Classification results saved to {output_file}")

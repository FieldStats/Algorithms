import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import re

def read_perfect_positions(file_path):
    """ Reads perfect positions from a formation_positions.txt file. """
    left_positions = {}
    current_team = None  

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Left Team Formation"):
                current_team = "left"

            match = re.match(r"Player (\d+): \(([\d.]+), ([\d.]+)\)", line)
            if match and current_team == "left":
                player_id = int(match.group(1))
                x, y = float(match.group(2)), float(match.group(3))
                left_positions[player_id] = (x, y)

    return left_positions

def generate_imperfect_positions(perfect_positions):
    """ Adds a random offset (1-10 units) to create imperfect positions. """
    imperfect_positions = {}
    for player_id, (x, y) in perfect_positions.items():
        x_offset = random.randint(1, 10)
        y_offset = random.randint(1, 10)
        imperfect_positions[player_id] = (x + x_offset, y + y_offset)
    return imperfect_positions

def compute_cost_matrix(perfect_positions, imperfect_positions):
    """ Computes the cost matrix based on Euclidean distance. """
    perfect_list = list(perfect_positions.values())
    imperfect_list = list(imperfect_positions.values())

    cost_matrix = np.zeros((len(perfect_list), len(imperfect_list)))

    for i, (x1, y1) in enumerate(perfect_list):
        for j, (x2, y2) in enumerate(imperfect_list):
            cost_matrix[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # Euclidean distance

    return cost_matrix

# ---- Load and generate positions ----
formation_file = "formation_positions.txt"
perfect_left = read_perfect_positions(formation_file)
imperfect_left = generate_imperfect_positions(perfect_left)
print(imperfect_left)

# Compute cost matrix and run Hungarian algorithm
cost_matrix = compute_cost_matrix(perfect_left, imperfect_left)
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Display results
print("\nHungarian Algorithm Matching Results:")
print("=====================================")
for i in range(len(row_ind)):
    perfect_id = list(perfect_left.keys())[row_ind[i]]
    imperfect_id = list(imperfect_left.keys())[col_ind[i]]
    perfect_pos = perfect_left[perfect_id]
    imperfect_pos = imperfect_left[imperfect_id]

    print(f"Perfect Player {perfect_id} {perfect_pos} → Matched with Imperfect Player {imperfect_id} {imperfect_pos}")


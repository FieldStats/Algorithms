import matplotlib.pyplot as plt
import math
    # --- NEW SECTION FOR get_formationCostDifference ---
import random
import numpy as np
   # ---- NEW: Distance-Based Cost Functions ----

def compute_distance_matrix(positions):
    """
    Computes a 10x10 matrix where each element [i][j] is the Euclidean distance
    between player i and player j.
    """
    n = len(positions)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
            else:
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                matrix[i][j] = math.sqrt(dx**2 + dy**2)
    return matrix

def parse_formation_positions():
    """
    Reads the perfect left/right team positions from formation_positions.txt.
    Returns two lists: (left_positions, right_positions).
    """
    left_positions = []
    right_positions = []
    current_team = None
    with open("formation_positions.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Left Team Formation:"):
                current_team = "left"
            elif line.startswith("Right Team Formation:"):
                current_team = "right"
            elif line.startswith("Player"):
                # Extract (x, y) from lines like "Player 1: (600, 75.0)"
                pos_str = line.split(": ")[1].strip("()")
                x, y = map(float, pos_str.split(", "))
                if current_team == "left":
                    left_positions.append((x, y))
                else:
                    right_positions.append((x, y))
    return left_positions, right_positions

def get_position_costdifference(imperfect_left, imperfect_right):
    """
    Computes cost matrices based on Euclidean distance differences between
    the input positions and the perfect formation's distance matrices.
    """
    # Get perfect positions from file
    perfect_left, perfect_right = parse_formation_positions()
    
    # Compute perfect distance matrices
    perfect_left_dist = compute_distance_matrix(perfect_left)
    perfect_right_dist = compute_distance_matrix(perfect_right)
    
    # Compute imperfect distance matrices
    imperfect_left_dist = compute_distance_matrix(imperfect_left)
    imperfect_right_dist = compute_distance_matrix(imperfect_right)
    
    # Calculate cost matrices (sum of squared differences)
    cost_left = np.zeros((10, 10))
    cost_right = np.zeros((10, 10))
    
    for i in range(10):
        for j in range(10):
            # Left team cost: compare imperfect[i] to perfect[j]
            diff_left = imperfect_left_dist[i] - perfect_left_dist[j]
            cost_left[i, j] = np.sum(diff_left ** 2)
            
            # Right team cost: compare imperfect[i] to perfect[j]
            diff_right = imperfect_right_dist[i] - perfect_right_dist[j]
            cost_right[i, j] = np.sum(diff_right ** 2)
    
    return cost_left, cost_right

def compute_angle_matrix(positions):
    """
    Computes an 10x10 matrix for the given list of positions.
    For each pair of players (i, j), the matrix element [i][j] is
    the smaller angle (in radians) between the 'north' direction (0, -1)
    and the vector from player i to player j.
    
    If i == j, the angle is set to 0.
    """
    n = len(positions)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
            else:
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dist = math.sqrt(dx**2 + dy**2)
                # To avoid division by zero just in case.
                if dist == 0:
                    angle = 0.0
                else:
                    # North is (0, -1). The dot product with (dx, dy) gives -dy.
                    cos_angle = -dy / dist
                    # Clamp the cosine value to the valid range [-1, 1].
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    angle = math.acos(cos_angle)
                matrix[i][j] = angle
    return matrix


def generate_positions(formation, team_side="left"):
    # Determine the number of players in each row based on the formation
    if formation == "4231":
        forwards, midfielders2, midfielders, defenders = 1, 3, 2, 4
    elif formation == "433":
        forwards, midfielders, defenders = 3, 3, 4
    elif formation == "4123":
        forwards, midfielders2, midfielders, defenders = 3, 2, 1, 4
    else:
        raise ValueError("Unsupported formation. Use '433' or '442'.")
    
    # Determine the x coordinate for each row based on the team side

    if formation == "4231" or formation == "4123": 
        if team_side == "left":
            forward_x = 600 * 4 // 4
            midfielder2_x = 600 * 3 // 4
            midfielder_x = 600 * 2 // 4
            defender_x = 600 // 4
        else:  # right side
            forward_x = 600 // 4
            midfielder2_x = 600 * 2 // 4
            midfielder_x = 600 * 3 // 4
            defender_x = 600 * 4 // 4

    if formation == "433":
        if team_side == "left":
            forward_x = 600 * 3 // 3
            midfielder_x = 600 * 2 // 3
            defender_x = 600 // 3
        else:  # right side
            forward_x = 600 // 3
            midfielder_x = 600 * 2 // 3
            defender_x = 600 * 3 // 3
    
    # Define the rows with their respective counts and x coordinates

    positions = []

    if formation == "4231" or formation == "4123": 
        rows = [
            (forwards, forward_x),
            (midfielders2, midfielder2_x),
            (midfielders, midfielder_x),
            (defenders, defender_x)
        ]
        
        for count, x in rows:
            if count == 0:
                continue  # Skip if no players in this row (though formations should have valid counts)
            # Compute y positions evenly spaced between 0 and 300
            if count == 1:
                ys = [150]  # Center vertically if only one player
            else:
                spacing = 300 / (count + 1)
                ys = [(i + 1) * spacing for i in range(count)]
            # Append each (x, y) position for all players in this row
            for y in ys:
                positions.append((x, y))

    if formation == "433":
        rows = [
            (forwards, forward_x),
            (midfielders, midfielder_x),
            (defenders, defender_x)
        ]
        
        for count, x in rows:
            if count == 0:
                continue  # Skip if no players in this row (though formations should have valid counts)
            # Compute y positions evenly spaced between 0 and 300
            if count == 1:
                ys = [150]  # Center vertically if only one player
            else:
                spacing = 300 / (count + 1)
                ys = [(i + 1) * spacing for i in range(count)]
            # Append each (x, y) position for all players in this row
            for y in ys:
                positions.append((x, y))
    
    return positions

import json

def save_perfect_matrices(formation_type_left="4123", formation_type_right="4231"):
    left_positions = generate_positions(formation_type_left, team_side="left")
    right_positions = generate_positions(formation_type_right, team_side="right")

    left_angle_matrix = compute_angle_matrix(left_positions)
    right_angle_matrix = compute_angle_matrix(right_positions)

    # Save angle matrices to a text file
    with open("formation_matrices.txt", "w") as f:
        f.write("Left Team Angle Matrix (radians):\n")
        for row in left_angle_matrix:
            f.write(" ".join(f"{angle:.4f}" for angle in row) + "\n")
        f.write("\nRight Team Angle Matrix (radians):\n")
        for row in right_angle_matrix:
            f.write(" ".join(f"{angle:.4f}" for angle in row) + "\n")

    # Save positions to a text file
    with open("formation_positions.txt", "w") as f:
        f.write(f"Left Team Formation: {formation_type_left}\n")
        for i, pos in enumerate(left_positions, start=1):
            f.write(f"Player {i}: {pos}\n")
        
        f.write(f"\nRight Team Formation: {formation_type_right}\n")
        for i, pos in enumerate(right_positions, start=1):
            f.write(f"Player {i}: {pos}\n")

    print("Perfect formation matrices saved to formation_matrices.txt")
    print("Player positions saved to formation_positions.txt")

def get_formationCostDifference(imperfect_left_positions, imperfect_right_positions):
    """
    Given imperfect positions for team 1 (left) and team 2 (right),
    this function reads the perfect 10x10 angle matrices from a text file
    ("formation_matrices.txt"), computes the imperfect angle matrices
    using compute_angle_matrix, and then calculates the element-wise absolute
    difference between perfect and imperfect matrices using the shorter angle rule.
    
    The overall cost for each team is the sum of these differences.
    """
    import math

    # Read the perfect angle matrices from file.
    perfect_left_matrix = []
    perfect_right_matrix = []
    with open("formation_matrices.txt", "r") as f:
        lines = f.readlines()
    
    # Locate the perfect left and right matrices in the file.
    left_start = None
    right_start = None
    for idx, line in enumerate(lines):
        if "Left Team Angle Matrix (radians):" in line:
            left_start = idx + 1
        if "Right Team Angle Matrix (radians):" in line:
            right_start = idx + 1
    
    if left_start is None or right_start is None:
        print("Error: Could not find perfect matrices in file.")
        return None

    # Assume the next 10 lines are the matrix rows.
    for i in range(10):
        row = lines[left_start + i].strip().split()
        perfect_left_matrix.append([float(val) for val in row])
    
    for i in range(10):
        row = lines[right_start + i].strip().split()
        perfect_right_matrix.append([float(val) for val in row])
    
    # Compute the imperfect angle matrices using your compute_angle_matrix function.
    imperfect_left_matrix = compute_angle_matrix(imperfect_left_positions)
    imperfect_right_matrix = compute_angle_matrix(imperfect_right_positions)
    
    result_left = np.zeros((10, 10))
    result_right = np.zeros((10, 10))

    imperfect_left_matrix = np.array(imperfect_left_matrix)
    perfect_left_matrix = np.array(perfect_left_matrix)

    imperfect_right_matrix = np.array(imperfect_right_matrix)
    perfect_right_matrix = np.array(perfect_right_matrix)
    # Compute the left result matrix.
    for i in range(10):
        for j in range(10):
            # Get the i-th row from imperfect_left_matrix and j-th row from perfect_left_matrix.
            diff_left = imperfect_left_matrix[i, :] - perfect_left_matrix[j, :]
            # Square the differences and sum them.
            result_left[i, j] = np.sum(diff_left  ** 2)

    # Compute the right result matrix.
    for i in range(10):
        for j in range(10):
            # Get the i-th row from imperfect_right_matrix and j-th row from perfect_right_matrix.
            diff_right = imperfect_right_matrix[i, :] - perfect_right_matrix[j, :]
            # Square the differences and sum them.
            result_right[i, j] = np.sum(diff_right ** 2)

    # Now, result_left and result_right are the two 10x10 matrices you need.
    print("Result Left Matrix:\n", result_left)
    print("Result Right Matrix:\n", result_right)
    # Calculate the cost difference matrix for each team.


    return result_left, result_right


if __name__ == "__main__":
    save_perfect_matrices()

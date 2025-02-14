import matplotlib.pyplot as plt
import math
    # --- NEW SECTION FOR get_formationCostDifference ---
import random
import numpy as np
   

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
    """
    Generates the positions for 10 players based on the given formation.
    
    Formation string is expected to be either "433" or "442", where the numbers represent:
        - defenders - midfielders - forwards.
    A goalkeeper (1 player) is added as the bottom row.
    
    For visualization, the rows are defined as follows (with y increasing downward):
        Row 0: Forwards
        Row 1: Midfielders
        Row 2: Defenders
        Row 3: Goalkeeper

    For the left team, positions in each row are ordered from left to right.
    For the right team, the positions are mirrored horizontally so that the ordering,
    when viewed from the team's perspective, is also left to right.
    """
    if formation == "433":
        defenders, midfielders, forwards = 4, 3, 3
    elif formation == "442":
        defenders, midfielders, forwards = 4, 4, 2
    else:
        raise ValueError("Unsupported formation. Use '433' or '442'.")
    
    # Define row order: forwards, midfielders, defenders, goalkeeper
    row_counts = [forwards, midfielders, defenders]
    # Assign y coordinates (row 0 is the top, row 3 is the bottom)
    y_values = [0, 1, 2, 3]
    
    positions = []
    for count, y in zip(row_counts, y_values):
        # Compute x positions evenly spaced between 0 and 1.
        if count == 1:
            xs = [0.5]
        else:
            xs = [i/(count-1) for i in range(count)]
        
        if team_side == "right":
            # Mirror the x coordinates and reverse the order so that in the team's
            # perspective, positions increase left-to-right.
            xs = [1 - x for x in xs]
            xs = list(reversed(xs))
        
        # Append positions (each as a tuple (x, y)) for all players in this row.
        for x in xs:
            positions.append((x, y))
            
    return positions


def save_perfect_matrices(formation_type_left= "433", formation_type_right= "433"):
    left_positions = generate_positions(formation_type_left, team_side="left")
    right_positions = generate_positions(formation_type_right, team_side="right")
    
    left_angle_matrix = compute_angle_matrix(left_positions)
    right_angle_matrix = compute_angle_matrix(right_positions)
    
    with open("formation_matrices.txt", "w") as f:
        f.write("Left Team Angle Matrix (radians):\n")
        for row in left_angle_matrix:
            f.write(" ".join(f"{angle:.4f}" for angle in row) + "\n")
        f.write("\nRight Team Angle Matrix (radians):\n")
        for row in right_angle_matrix:
            f.write(" ".join(f"{angle:.4f}" for angle in row) + "\n")
    print("Perfect formation matrices saved to formation_matrices.txt")

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
            result_left[i, j] = np.sum(diff_left ** 2)

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

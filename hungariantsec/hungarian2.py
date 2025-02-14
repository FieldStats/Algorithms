import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import formations 
def get_player_positions_next(data, split_frame):
    """
    Returns a dictionary mapping each player id (int) to its position (x, y) at the split frame.
    
    For each player (skipping object id 23), this function checks the player's paths:
      - If the split frame falls within a path (i.e. between the path's start and end frame),
        the corresponding position in that path is returned.
      - Otherwise, if no path contains the split frame, it looks for the next path (one with
        a start frame greater than the split frame) and returns its first position.
    
    Parameters:
        data (dict): The data containing player objects and their paths.
        split_frame (int): The frame at which to retrieve the position.
        
    Returns:
        dict: A dictionary mapping player ids (int) to their [x, y] positions.
    """
    positions = {}
    for obj_id, obj in data.items():

        chosen_position = None
        candidate_path = None
        candidate_start_frame = None

        for path in obj['paths']:
            # Determine the start and end frame for this path.
            start_frame = path['last_seen_frame'] - (len(path['path']) - 1)
            end_frame = path['last_seen_frame']
            
            # If the split frame falls within this path, choose that position.
            if start_frame <= split_frame <= end_frame:
                frame_offset = split_frame - start_frame
                chosen_position = path['path'][frame_offset]
                break  # Use the first matching path.
            
            # Otherwise, if the split frame is before the path starts, consider it as a candidate.
            if split_frame < start_frame:
                # Choose the candidate with the smallest start_frame (i.e. earliest upcoming path).
                if candidate_start_frame is None or start_frame < candidate_start_frame:
                    candidate_start_frame = start_frame
                    candidate_path = path

        # If no path contained the split frame, use the next path's first position (if available).
        if chosen_position is None and candidate_path is not None:
            chosen_position = candidate_path['path'][0]
            
        if chosen_position:
            positions[int(obj_id)] = chosen_position
            
    return positions


def get_player_positions(data, split_frame):
    """
    Returns a dictionary mapping each player id (int) to its position (x, y) at the split frame.
    (Skips object id 23, e.g. referee.)
    """
    positions = {}
    for obj_id, obj in data.items():
       
        latest_position = None
        for path in obj['paths']:
            start_frame = path['last_seen_frame'] - (len(path['path']) - 1)
            end_frame = path['last_seen_frame']
            
            if start_frame <= split_frame <= end_frame:
                frame_offset = split_frame - start_frame
                latest_position = path['path'][frame_offset]
                break
                
            if path['last_seen_frame'] < split_frame:
                latest_position = path['path'][-1]
                
        if latest_position:
            positions[int(obj_id)] = latest_position
    return positions


def get_extreme_positions(data, split_frame):
    """
    Uses get_player_positions to determine which object is the leftmost (smallest x)
    and which is the rightmost (largest x) at the split frame.
    Returns a dict with keys "left" and "right" mapping to object ids.
    """
    positions = get_player_positions(data, split_frame)
    
    leftmost_id = None
    rightmost_id = None
    leftmost_x = float('inf')
    rightmost_x = float('-inf')
    
    for obj_id, pos in positions.items():
        x, _ = pos
        if x < leftmost_x:
            leftmost_x = x
            leftmost_id = obj_id
        if x > rightmost_x:
            rightmost_x = x
            rightmost_id = obj_id

    return {"left": leftmost_id, "right": rightmost_id}


def get_player_team(data, split_frame):
    """
    Returns a dictionary mapping each player id (int) to its team (from the selected path)
    at the split frame.
    
    The same logic is used as in get_player_positions:
      - If the split_frame falls within a path's frame range, then that path's team is used.
      - Otherwise, if the path ended before split_frame, the team from the last position is used.
      
    Object id 23 is skipped.
    """
    teams = {}
    for obj_id, obj in data.items():
      #  if int(obj_id) == 23:
        #    continue

        team_val = None
        for path in obj['paths']:
            start_frame = path['last_seen_frame'] - (len(path['path']) - 1)
            end_frame = path['last_seen_frame']
            
            if start_frame <= split_frame <= end_frame:
                team_val = path.get('team_index', None)
                break
            
            if path['last_seen_frame'] < split_frame:
                team_val = path.get('team_index', None)
                
        if team_val is not None:
            teams[int(obj_id)] = team_val
    print(teams)
    return teams


def classify_non_extreme_players(data, split_frame):
    """
    For the 21 non-extreme players (i.e. excluding the leftmost and rightmost players),
    classify them into three teams based on the team value from their split-frame path.
    
    The expected grouping is:
      - Team -1: 10 players
      - Team  0:  1 player
      - Team  1: 10 players
    
    If the initial grouping does not match the expected counts (i.e. one group is off by one),
    one extra (overflowing) element from the surplus group is swapped into the deficit group.
    
    Returns a dict with keys -1, 0, and 1 mapping to lists of object ids.
    """
    # Get the extreme players (these will be excluded)
    extremes = get_extreme_positions(data, split_frame)
    left_extreme = extremes.get("left")
    right_extreme = extremes.get("right")
    
    # Get team values for all players at the split frame.
    teams_all = get_player_team(data, split_frame)
    # Exclude the extreme players
    non_extreme_teams = {pid: team for pid, team in teams_all.items() if pid not in [left_extreme, right_extreme]}
    

    print(teams_all)
    print(non_extreme_teams)   

    if len(non_extreme_teams) != 21:
        raise ValueError(f"Expected 21 non-extreme players, got {len(non_extreme_teams)}")
    
    # Group players by their team value.
    team_groups = { -1: [], 0: [], 1: [] }
    for pid, team_val in non_extreme_teams.items():
        if team_val in team_groups:
            team_groups[team_val].append(pid)
        else:
            # In case an unexpected team value is encountered, you might decide to handle it.
            pass

    # Expected counts:
    expected_counts = { -1: 10, 0: 1, 1: 10 }

    if len(team_groups[0]) == 0:
        max_team = max(team_groups, key=lambda t: len(team_groups[t]) if t != 0 else -1)
        if team_groups[max_team]:  # Ensure the team has at least one player to move
            player_to_move = team_groups[max_team].pop()
            team_groups[0].append(player_to_move)

    diff = { team: len(team_groups[team]) - expected_counts[team] for team in team_groups }

    # Continue swapping until all differences are balanced (i.e. diff is 0 for every team)
    while any(d > 0 for d in diff.values()) and any(d < 0 for d in diff.values()):
        # Identify teams with surplus and with deficit (allowing for differences larger than 1)
        surplus_teams = [team for team, d in diff.items() if d > 0]
        deficit_teams = [team for team, d in diff.items() if d < 0]
        
        # Choose one surplus and one deficit team (you could also iterate over all pairs if desired)
        surplus_team = surplus_teams[0]
        deficit_team = deficit_teams[0]
        
        # Move one element from the surplus team to the deficit team.
        if team_groups[surplus_team]:  # ensure there's at least one element to pop
            element_to_swap = team_groups[surplus_team].pop()
            team_groups[deficit_team].append(element_to_swap)
            
            # Update the differences.
            diff[surplus_team] -= 1
            diff[deficit_team] += 1
    
    print(expected_counts)
    # Final check that counts match the expected numbers.
    for team in expected_counts:
        if len(team_groups[team]) != expected_counts[team]:
            raise ValueError(
                f"After swapping, team {team} has {len(team_groups[team])} players; expected {expected_counts[team]}."
            )
    
    return team_groups

import numpy as np

def compute_id_cost_matrix(valid_ids):
    """
    Computes an n x n matrix of ID costs.
    
    For each pair (i, j) in valid_ids, the cost is defined as:
         id_cost = abs(int(valid_ids[i]) - int(valid_ids[j])) * id_weight
    
    Args:
        valid_ids (list): A list of player IDs (as strings or integers).
        id_weight (float): The weight applied to the absolute ID difference.
    
    Returns:
        np.ndarray: An n x n numpy array containing the ID cost for each pairing.
    """
    n = len(valid_ids)
    id_cost_matrix = np.zeros((n, n))
    for i, obj_id in enumerate(valid_ids):
        for j, target_id in enumerate(valid_ids):
            id_cost_matrix[i][j] = abs(int(obj_id) - int(target_id)) 
    return id_cost_matrix



def compute_distance_matrix(player_positions):
    """Compute the 11x11 Euclidean distance matrix for a given team's positions."""
    ids = sorted(player_positions.keys())  # Ensure ordered by ID
    n = len(ids)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:  # Avoid self-distance
                pos_i = player_positions[ids[i]]
                pos_j = player_positions[ids[j]]
                distance_matrix[i, j] = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
    
    return distance_matrix

def final_index_matching(teams, player_positions,
                         prev_left_extreme, next_left_extreme,
                         prev_right_extreme, next_right_extreme,
                         fixed_self):
    """
    Combines two Hungarian matchings with fixed assignments to produce a final matching of 23 pairs.
    
    Input assumptions:
      - For the first Hungarian matching, we use:
          • teams[1] : list of 10 player IDs from snapshot1 (team1)
          • teams[-1]: list of 10 player IDs from snapshot1 (team–1)
      - For the second Hungarian matching, we use:
          • teams[2] : list of 10 player IDs from snapshot2 (team1)
          • teams[-2]: list of 10 player IDs from snapshot2 (team–1)
          
      - The fixed assignments are provided as follows (and are disjoint from the Hungarian matching IDs):
          • Left extreme: (prev_left_extreme, next_left_extreme)
          • Right extreme: (prev_right_extreme, next_right_extreme)
          • Self-match: (fixed_self, fixed_self)
    
    The final matching will be a list of 23 pairs (tuples) of the form:
         (team1_snapshot_ID, team_minus1_snapshot_ID)
    and will be printed out.
    """
    
    # --- Hungarian Matching 1 (from snapshot1) ---
    # Compute cost matrix for snapshot1 using the team IDs provided.
    # Note: The order of arguments in get_formationCostDifference is assumed such that:
    #       cost1 is used to match team1 (rows, from teams[1]) with team–1 (cols, from teams[-1]).
    team1_positions = {player_id: player_positions[player_id] for player_id in teams[1] if player_id in player_positions}
    team_minus1_positions = {player_id: player_positions[player_id] for player_id in teams[-1] if player_id in player_positions}

    cost1, cost2 = formations.get_formationCostDifference(
                    compute_id_cost_matrix(team_minus1_positions),
                    compute_id_cost_matrix(team1_positions))
    row_ind1, col_ind1 = linear_sum_assignment(cost1)
    row_ind2, col_ind2 = linear_sum_assignment(cost2)
    # Retrieve the actual player IDs for snapshot1 from the Hungarian matching.
    hungarian1_team_minus1 = [teams[-1][r] for r in row_ind1]      
    hungarian1_team_minus1_matched = [teams[-1][r] for r in col_ind1]    
    hungarian2_team_1 = [teams[1][r] for r in row_ind2]      
    hungarian2_team_1_matched = [teams[1][r] for r in col_ind2]    

    final_match_left = hungarian1_team_minus1 + hungarian2_team_1 + [prev_left_extreme, prev_right_extreme, fixed_self]
    final_match_right = hungarian1_team_minus1_matched + hungarian2_team_1_matched + [next_left_extreme, next_right_extreme, fixed_self]

    if len(final_match_left) != 23 or len(final_match_right) != 23:
        raise ValueError("Final matching does not have 23 pairs!")

    # Print the final matching pairs:
    print("Final Matching (Total 23 pairs):")
    for i in range(23):
        print(f"Pair {i+1}: Team1 ID {final_match_left[i]}  <-->  Team–1 ID {final_match_right[i]}")
    
    return final_match_left, final_match_right



def process_matching(input_file, output_file, t=2, fps=60,
                    team_weight=0.0, formation_weight=0.0, id_weight=1.0):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    split_interval = t * fps
    all_ids = sorted(data.keys(), key=lambda x: int(x))  # Maintain ID order
    
    max_frame = max(p['last_seen_frame'] for obj in data.values() for p in obj['paths'])
    split_frames = [i * split_interval for i in range(1, (max_frame // split_interval) + 1)]

    for split_frame in split_frames:
        print(f"\nProcessing split frame {split_frame}")
        
        # Find closest paths around split frame for all IDs
        prev_paths = {}
        next_paths = {}
        
        for obj_id in all_ids:
            obj = data[obj_id]
            paths = obj['paths']
            
            # Find closest previous path (last_seen <= split_frame)
            prev_candidates = [p for p in paths if p['last_seen_frame'] <= split_frame]
            prev = max(prev_candidates, key=lambda x: x['last_seen_frame']) if prev_candidates else None
            
            # Find closest next path (starts after split_frame)
            next_candidates = []
            for p in paths:
                start_frame = p['last_seen_frame'] - (len(p['path']) - 1)
                if start_frame > split_frame:
                    next_candidates.append((start_frame, p))
            
            nxt = min(next_candidates, key=lambda x: x[0])[1] if next_candidates else None
            
            prev_paths[obj_id] = prev
            next_paths[obj_id] = nxt

        # Create cost matrix with weighted components
        valid_ids = [obj_id for obj_id in all_ids if prev_paths[obj_id] and next_paths[obj_id]]
        n = len(valid_ids)
        print("n :\n", n) 
        if n == 0:
            continue
        if n != 23:
            continue

        #cost_matrix += compute_id_cost_matrix(valid_ids, id_weight)
        player_positions = get_player_positions(data, split_frame)
        player_positions_next = get_player_positions_next(data, split_frame)
        #print(len(player_positions))
        #print(len(player_positions_next))

        prev_extremes = get_extreme_positions(data, split_frame)
        prev_left_extreme = prev_extremes.get("left")
        prev_right_extreme = prev_extremes.get("right")
        next_extremes = get_extreme_positions(data, split_frame)
        next_left_extreme = next_extremes.get("left")
        next_right_extreme = next_extremes.get("right")

        teams = classify_non_extreme_players(data, split_frame)
        fixed_self = 16

        final_left, final_right = final_index_matching(
        teams, player_positions,
        prev_left_extreme, next_left_extreme,
        prev_right_extreme, next_right_extreme,
        fixed_self
        )

        print("Team -1:", teams[-1])
        print("Team  0:", teams[0])
        print("Team  1:", teams[1])
        
        # Create ID mapping based on matches
        id_mapping = {}
        for i, j in zip(final_left, final_right):
            original_id = str(i)
            target_id   = str(j)
            id_mapping[original_id] = target_id

        # 2. Print the mappings to the console for clarity
        print(f"Mapped IDs at frame {split_frame}:")
        for orig, new in id_mapping.items():
            print(f"{orig} → {new}")

        # 3. Prepare the JSON structure
        mapping_data = {
            "frame": split_frame,
            "mapping": id_mapping
        }

    # 4. Save to JSON (no changes made to original data)
    with open(output_file, 'w') as f:
        json.dump(mapping_data, f, indent=4)
    print(f"\nMapping saved to {output_file}")

# Example usage with custom weights
process_matching('output.json', 'output2.json', t=5, fps=60,
                 team_weight=0.5,   # Currently not used (multiplies 0)
                 formation_weight=0.2,  # Currently not used
                 id_weight=0.8)
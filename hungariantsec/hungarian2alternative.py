import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import formations 

def get_player_positions_next(data, split_frame):
    """
    Returns a dictionary mapping each player id (int) to its position (x, y) at the split frame.
    For each player (skipping object id 23), this function checks the player's paths:
      - If the split frame falls within a path, the corresponding position is returned.
      - Otherwise, if no path contains the split frame, it looks for the next path and returns its first position.
    """
    positions = {}
    for obj_id, obj in data.items():
        chosen_position = None
        candidate_path = None
        candidate_start_frame = None
        for path in obj['paths']:
            start_frame = path['last_seen_frame'] - (len(path['path']) - 1)
            end_frame = path['last_seen_frame']
            if start_frame <= split_frame <= end_frame:
                frame_offset = split_frame - start_frame
                chosen_position = path['path'][frame_offset][1:]
                break
            if split_frame < start_frame:
                if candidate_start_frame is None or start_frame < candidate_start_frame:
                    candidate_start_frame = start_frame
                    candidate_path = path
        if chosen_position is None and candidate_path is not None:
            chosen_position = candidate_path['path'][0][1:]
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
                latest_position = path['path'][frame_offset][1:]
                break
            if path['last_seen_frame'] < split_frame:
                latest_position = path['path'][-1][1:]
        if latest_position:
            positions[int(obj_id)] = latest_position
    return positions


def get_extreme_positions(data, split_frame):
    """
    Determines which object is the leftmost (smallest x) and rightmost (largest x)
    at the split frame, using get_player_positions.
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
    """
    teams = {}
    for obj_id, obj in data.items():
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
    For the 21 non-extreme players (excluding the leftmost and rightmost players),
    classify them into three teams based on the team value from their split-frame path.
    """
    extremes = get_extreme_positions(data, split_frame)
    left_extreme = extremes.get("left")
    right_extreme = extremes.get("right")
    teams_all = get_player_team(data, split_frame)
    non_extreme_teams = {pid: team for pid, team in teams_all.items() if pid not in [left_extreme, right_extreme]}
    print(teams_all)
    print(non_extreme_teams)
    if len(non_extreme_teams) != 21:
        raise ValueError(f"Expected 21 non-extreme players, got {len(non_extreme_teams)}")
    team_groups = { -1: [], 0: [], 1: [] }
    for pid, team_val in non_extreme_teams.items():
        if team_val in team_groups:
            team_groups[team_val].append(pid)
    expected_counts = { -1: 10, 0: 1, 1: 10 }
    if len(team_groups[0]) == 0:
        max_team = max(team_groups, key=lambda t: len(team_groups[t]) if t != 0 else -1)
        if team_groups[max_team]:
            player_to_move = team_groups[max_team].pop()
            team_groups[0].append(player_to_move)
    diff = { team: len(team_groups[team]) - expected_counts[team] for team in team_groups }
    while any(d > 0 for d in diff.values()) and any(d < 0 for d in diff.values()):
        surplus_teams = [team for team, d in diff.items() if d > 0]
        deficit_teams = [team for team, d in diff.items() if d < 0]
        surplus_team = surplus_teams[0]
        deficit_team = deficit_teams[0]
        if team_groups[surplus_team]:
            element_to_swap = team_groups[surplus_team].pop()
            team_groups[deficit_team].append(element_to_swap)
            diff[surplus_team] -= 1
            diff[deficit_team] += 1
    print(expected_counts)
    for team in expected_counts:
        if len(team_groups[team]) != expected_counts[team]:
            raise ValueError(
                f"After swapping, team {team} has {len(team_groups[team])} players; expected {expected_counts[team]}."
            )
    return team_groups


def compute_id_cost_matrix(valid_ids):
    """
    Computes an n x n matrix of ID costs.
    """
    n = len(valid_ids)
    id_cost_matrix = np.zeros((n, n))
    for i, obj_id in enumerate(valid_ids):
        for j, target_id in enumerate(valid_ids):
            id_cost_matrix[i][j] = abs(int(obj_id) - int(target_id))
    return id_cost_matrix


def compute_distance_matrix(player_positions):
    """
    Compute the Euclidean distance matrix for a given team's positions.
    """
    ids = sorted(player_positions.keys())
    n = len(ids)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
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
    """
    team1_positions = {player_id: player_positions[player_id] for player_id in teams[1] if player_id in player_positions}
    team_minus1_positions = {player_id: player_positions[player_id] for player_id in teams[-1] if player_id in player_positions}

    cost1, cost2 = formations.get_formationCostDifference(
                    compute_id_cost_matrix(team_minus1_positions),
                    compute_id_cost_matrix(team1_positions))
    row_ind1, col_ind1 = linear_sum_assignment(cost1)
    row_ind2, col_ind2 = linear_sum_assignment(cost2)
    hungarian1_team_minus1 = [teams[-1][r] for r in row_ind1]      
    hungarian1_team_minus1_matched = [teams[-1][r] for r in col_ind1]    
    hungarian2_team_1 = [teams[1][r] for r in row_ind2]      
    hungarian2_team_1_matched = [teams[1][r] for r in col_ind2]    

    final_match_left = hungarian1_team_minus1 + hungarian2_team_1 + [prev_left_extreme, prev_right_extreme, fixed_self]
    final_match_right = hungarian1_team_minus1_matched + hungarian2_team_1_matched + [next_left_extreme, next_right_extreme, fixed_self]

    if len(final_match_left) != 23 or len(final_match_right) != 23:
        raise ValueError("Final matching does not have 23 pairs!")

    print("Final Matching (Total 23 pairs):")
    for i in range(23):
        print(f"Pair {i+1}: Team1 ID {final_match_left[i]}  <-->  Team–1 ID {final_match_right[i]}")
    
    return final_match_left, final_match_right


def process_matching(input_file, mapping_output_file, positions_output_file, t=2, fps=60):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    split_interval = t * fps
    all_ids = sorted(data.keys(), key=lambda x: int(x))
    max_frame = max(p['last_seen_frame'] for obj in data.values() for p in obj['paths'])
    split_frames = [i * split_interval for i in range(1, (max_frame // split_interval) + 1)]
    
    all_mappings = []
    all_positions = []  # New list to hold positions for each split frame

    for split_frame in split_frames:
        print(f"\nProcessing split frame {split_frame}")
        prev_paths = {}
        next_paths = {}
        for obj_id in all_ids:
            obj = data[obj_id]
            paths = obj['paths']
            prev_candidates = [p for p in paths if p['last_seen_frame'] <= split_frame]
            prev = max(prev_candidates, key=lambda x: x['last_seen_frame']) if prev_candidates else None
            next_candidates = []
            for p in paths:
                start_frame = p['last_seen_frame'] - (len(p['path']) - 1)
                if start_frame > split_frame:
                    next_candidates.append((start_frame, p))
            nxt = min(next_candidates, key=lambda x: x[0])[1] if next_candidates else None
            prev_paths[obj_id] = prev
            next_paths[obj_id] = nxt

        valid_ids = [obj_id for obj_id in all_ids if prev_paths[obj_id] and next_paths[obj_id]]
        n = len(valid_ids)
        print("n :\n", n) 
        if n == 0:
            continue
        if n != 23:
            continue
        if split_frame > 9000:
            continue

        player_positions = get_player_positions(data, split_frame)
        player_positions_next = get_player_positions_next(data, split_frame)

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
        
        id_mapping = {}
        for i, j in zip(final_left, final_right):
            original_id = str(i)
            target_id   = str(j)
            id_mapping[original_id] = target_id

        print(f"Mapped IDs at frame {split_frame}:")
        for orig, new in id_mapping.items():
            print(f"{orig} → {new}")

        mapping_data = {
            "frame": split_frame,
            "mapping": id_mapping
        }
        all_mappings.append(mapping_data)
        
        # Save player positions for this frame to a separate structure.
        positions_entry = {
            "frame": split_frame,
            "positions": player_positions_next
        }
        all_positions.append(positions_entry)

    with open(mapping_output_file, 'w') as f:
        json.dump(all_mappings, f, indent=4)
    print(f"\nMapping saved to {mapping_output_file}")
    
    # Save the positions from each split frame into a separate file.
    with open(positions_output_file, 'w') as f:
        json.dump(all_positions, f, indent=4)
    print(f"Player positions saved to {positions_output_file}")


# Example usage:
process_matching('output.json', 'output2test.json', 'positions.json', t=5, fps=60)

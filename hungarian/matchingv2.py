#!/usr/bin/env python3

import json
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import os

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    # Read JSON from file in current directory
    json_path = os.path.join(os.getcwd(), "filtered_hungarian.json")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

    # Extract tracklets dictionary
    # data should look like: {"tracklets": {...}}
    tracklets = data.get("tracklets", {})
    if not tracklets:
        print("No tracklets found in the JSON!")
        sys.exit(0)

    # 1) Identify the tracklets that start at 404
    #    (In your real case, you expect 23. In the test example, maybe 2.)
    start_404_ids = [tid for tid, tdata in tracklets.items() if tdata.get('start_time') == 404]

    # If you want to warn if it's not 23, do so:
    if len(start_404_ids) != 23:
        print(f"WARNING: Expected exactly 23 tracklets with start_time=404, found {len(start_404_ids)}!")

    # Initialize one chain per "start_time=404" tracklet
    chains = [[tid] for tid in start_404_ids]

    # Helper to get the last tracklet ID in a chain
    def last_tid(chain):
        return chain[-1]

    # Helper to get end_time of the last tracklet in a chain
    def chain_end_time(chain):
        return tracklets[last_tid(chain)]['end_time']

    changed = True
    while changed:
        changed = False

        # Sort chains by their last tracklet's end_time (earliest first)
        chains.sort(key=chain_end_time)

        # Collect all chains whose last tracklet has a valid next_wrong
        # (We won't pre-check if it's time-feasible; let the cost matrix handle infinite cost.)
        eligible_chains = []
        candidate_tracklets = set()

        for ch in chains:
            t_end_id = last_tid(ch)
            t_end_data = tracklets[t_end_id]
            nw_id = t_end_data.get('next_wrong')
            if nw_id is not None and nw_id in tracklets:
                eligible_chains.append(ch)
                candidate_tracklets.add(nw_id)

        # If no chain is eligible to expand, we're done.
        if not eligible_chains:
            break

        # Convert candidate tracklets set to list for indexing
        candidate_tracklets_list = list(candidate_tracklets)

        # Build the cost matrix: rows = eligible_chains, cols = candidate tracklets
        num_rows = len(eligible_chains)
        num_cols = len(candidate_tracklets_list)
        cost_matrix = np.zeros((num_rows, num_cols), dtype=float)

        for i, ch in enumerate(eligible_chains):
            t_end_id = last_tid(ch)
            t_end_data = tracklets[t_end_id]
            end_time = t_end_data['end_time']
            x_end = t_end_data['x_end']
            y_end = t_end_data['y_end']

            for j, cand_id in enumerate(candidate_tracklets_list):
                cand_data = tracklets[cand_id]
                start_time = cand_data['start_time']
                x_start = cand_data['x_start']
                y_start = cand_data['y_start']

                # If candidate starts too early, cost is infinite
                if start_time < end_time:
                    cost_matrix[i, j] = float('inf')
                else:
                    # Otherwise cost = Euclidean distance
                    dist = euclidean_distance(x_end, y_end, x_start, y_start)
                    cost_matrix[i, j] = dist

        # Use Hungarian (Munkres) to get minimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # For each assigned pair
        used_candidates = set()
        for r, c in zip(row_ind, col_ind):
            this_cost = cost_matrix[r, c]
            if this_cost == float('inf'):
                print("WARNING: forced an infinite cost match!")
                # We could skip forcibly invalid matches if desired
                continue

            chosen_chain = eligible_chains[r]
            chosen_cand_id = candidate_tracklets_list[c]

            # Just link them if cost is finite
            # (No check that `chosen_cand_id` == chain's last_tracklet.next_wrong)
            if chosen_cand_id not in chosen_chain:
                chosen_chain.append(chosen_cand_id)
                changed = True

            used_candidates.add(chosen_cand_id)

        # End while-loop iteration: if changed==True, we continue; else we stop

    # All expansions done. Print final chains
    print("=== Final Chains ===")
    for i, chain in enumerate(chains, start=1):
        if not chain:
            continue
        first_tid = chain[0]
        last_id = chain[-1]
        start_t = tracklets[first_tid]['start_time']
        end_t   = tracklets[last_id]['end_time']
        print(f"Chain {i}: start={start_t}, end={end_t}")

if __name__ == "__main__":
    main()

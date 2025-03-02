import json

def process_data(input_file, output_file, t=2, fps=60, frame_offset=0):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    split_interval = t * fps  # Calculate frames per interval

    for obj_key in data:
        obj = data[obj_key]
        paths = obj['paths']
        if not paths:
            continue
        
        # Determine the split frames for this object
        max_last_seen = max(p['last_seen_frame'] for p in paths)
        split_frames = []
        current_split = split_interval
        while current_split <= max_last_seen:
            split_frames.append(current_split)
            current_split += split_interval
        
        current_paths = paths.copy()
        
        # Process each split frame in ascending order
        for S in split_frames:
            new_paths = []
            for path in current_paths:
                path_len = len(path['path'])
                if path_len == 0:
                    new_paths.append(path)
                    continue
                
                last_seen = path['last_seen_frame']
                team = path['team_index']
                start_frame = last_seen - (path_len - 1)
                end_frame = last_seen
                
                if start_frame < S <= end_frame:
                    split_pos = S - start_frame
                    # Split the path into two parts
                    part1 = {
                        'id': path['id'],
                        'path': path['path'][:split_pos],
                        'last_seen_frame': start_frame + split_pos - 1,
                        'team_index': team
                    }
                    part2 = {
                        'id': path['id'],
                        'path': path['path'][split_pos:],
                        'last_seen_frame': end_frame,
                        'team_index': team
                    }
                    print(start_frame)
                    print(end_frame)
                    print(last_seen)
                    
                    new_paths.append(part1)
                    new_paths.append(part2)
                    print(f"Split path for id {path['id']} at frame {S} into two parts: new last_seen frames {part1['last_seen_frame']} and {part2['last_seen_frame']}.")
                else:
                    new_paths.append(path)
            
            current_paths = new_paths
        
        # Apply frame offset to all paths
        for path in current_paths:
            path['last_seen_frame'] = max(0, path['last_seen_frame'] - frame_offset)
        
        obj['paths'] = current_paths
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
input_file = 'debug3.json'
output_file = 'output.json'
t = 5
fps = 60
frame_offset = 0  # Adjust as needed

process_data(input_file, output_file, t, fps, frame_offset)
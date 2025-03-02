import json

def substitute(num, prog_map):
    """
    Substitute a number using prog_map.
    If num is a key in prog_map, return its mapped value;
    otherwise, return num unchanged.
    """
    return prog_map.get(num, num)

def update_program_map(prog_map, new_mapping):
    """
    For each pair (A -> B) in new_mapping,
    update prog_map by replacing every occurrence of B (in keys and values) with A.
    """
    for A, B in new_mapping.items():
        updated_map = {}
        for key, val in prog_map.items():
            # Replace key if it matches B, otherwise keep it
            new_key = A if key == B else key
            # Replace value if it matches B, otherwise keep it
            new_val = A if val == B else val
            updated_map[new_key] = new_val
        prog_map = updated_map
    return prog_map

def process_frames(frames):
    """
    Process the list of frames according to the substitution and program map update steps.
    The first frame is used only to create the program map (by reversing its mapping)
    and is output unchanged.
    Returns a new list of frames (dictionaries) with updated mappings.
    """
    output_frames = []
    program_map = {}
    
    if frames:
        # Use first frame to initialize the program map.
        first_frame = frames[0]
        input_map = first_frame['mapping']
        # Create program map by reversing the input mapping.
        program_map = {value: key for key, value in input_map.items()}
        # Do not update the first frame output; output it as is.
        output_frames.append({
            'frame': first_frame['frame'],
            'mapping': input_map.copy()
        })
    
    # Process subsequent frames (frame 2 and onward)
    for frame in frames[1:]:
        input_map = frame['mapping']
        new_mapping = {}
        # Substitution: for each pair, substitute both key and value using program_map.
        for k, v in input_map.items():
            new_k = substitute(k, program_map)
            new_v = substitute(v, program_map)
            new_mapping[new_k] = new_v
        
        # Update the program map with the new mapping.
        program_map = update_program_map(program_map, new_mapping)
        
        # Append the transformed mapping as the output for this frame.
        output_frames.append({
            'frame': frame['frame'],
            'mapping': new_mapping.copy()
        })
    
    return output_frames

def main():
    # Read input from "output2.json"
    with open("output2.json", "r") as infile:
        frames = json.load(infile)
    
    # Process frames using our substitution and update algorithm.
    output_frames = process_frames(frames)
    
    # Write the output to "output2.5.json"
    with open("output2.5.json", "w") as outfile:
        json.dump(output_frames, outfile, indent=4)

if __name__ == "__main__":
    main()

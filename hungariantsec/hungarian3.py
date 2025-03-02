import json
import copy

def split_segment(segment, frame):
    """
    Splits a segment at the given frame if the segment spans the boundary.
    
    Each segment has a "last_seen_frame" and a "path" list.
    The starting frame of the segment is computed as:
    
        start_frame = last_seen_frame - len(path) + 1
        
    If the segment spans the mapping frame, we split it into two segments:
      - The 'before' part covers frames [start_frame, frame-1]
      - The 'after' part covers frames [frame, last_seen_frame]
    
    When computing the 'after' segment, we effectively subtract the number
    of path points that belong to frames before the mapping frame.
    
    Returns a tuple (seg_before, seg_after) where either value may be None if
    the segment lies entirely on one side of the frame.
    """
    path = segment["path"]
    total_frames = len(path)
    start_frame = segment["last_seen_frame"] - total_frames + 1
    end_frame = segment["last_seen_frame"]
    
    # If the entire segment ends before the mapping frame:
    if end_frame < frame:
        return (segment, None)
    # If the entire segment starts at or after the mapping frame:
    if start_frame >= frame:
        return (None, segment)
    
    # The segment spans the mapping frame. Determine how many frames belong to the 'before' part.
    frames_before = frame - start_frame  # number of path points before the mapping frame
    frames_before = max(0, min(frames_before, total_frames))
    
    seg_before = None
    if frames_before > 0:
        seg_before = copy.deepcopy(segment)
        seg_before["path"] = path[:frames_before]
        seg_before["last_seen_frame"] = start_frame + frames_before - 1  # should equal frame - 1

    seg_after = None
    frames_after = total_frames - frames_before
    if frames_after > 0:
        seg_after = copy.deepcopy(segment)
        seg_after["path"] = path[frames_before:]
        seg_after["last_seen_frame"] = frame + frames_after - 1

    return (seg_before, seg_after)

def partition_segments(segments, frame):
    """
    Given a list of segments and a mapping frame, partitions them into two lists:
      - before_segments: segments (or segment parts) entirely before the frame
      - after_segments: segments (or segment parts) that start at or after the frame
    
    If a segment spans the boundary, it is split into two parts.
    """
    before_segments = []
    after_segments = []
    for seg in segments:
        total_frames = len(seg["path"])
        start_frame = seg["last_seen_frame"] - total_frames + 1
        end_frame = seg["last_seen_frame"]

        if end_frame < frame:
            before_segments.append(seg)
        elif start_frame >= frame:
            after_segments.append(seg)
        else:
            seg_before, seg_after = split_segment(seg, frame)
            if seg_before is not None:
                before_segments.append(seg_before)
            if seg_after is not None:
                after_segments.append(seg_after)
    return before_segments, after_segments

def main():
    # Load the JSON data.
    with open("output.json", "r") as f:
        data = json.load(f)
    
    with open("output2.5.json", "r") as f:
        mapping_list = json.load(f)
    
    # Sort mapping events in ascending order of the frame.
    mapping_list.sort(key=lambda x: x["frame"])
    
    # Process each mapping event.
    for mapping_obj in mapping_list:
        frame = mapping_obj["frame"]
        mapping = mapping_obj["mapping"]
        processed_pairs = set()  # To avoid processing the same pair twice.
        
        # Process every mapping pair, regardless of reciprocity.
        for id_a, partner in mapping.items():
            pair = tuple(sorted([id_a, partner]))
            if pair in processed_pairs:
                continue
            
            if id_a not in data or partner not in data:
                print(f"Warning: Main track {id_a} or {partner} not found in output.json.")
                continue
            
            # Partition segments for each track relative to the mapping frame.
            segs_a = data[id_a].get("paths", [])
            segs_partner = data[partner].get("paths", [])
            before_a, after_a = partition_segments(segs_a, frame)
            before_partner, after_partner = partition_segments(segs_partner, frame)
            
            # Merge segments:
            # For track id_a, keep its 'before' segments plus partner’s 'after' segments.
            new_segs_a = before_a + after_partner
            # For the partner, keep its 'before' segments plus id_a’s 'after' segments.
            new_segs_partner = before_partner + after_a
            
            data[id_a]["paths"] = new_segs_a
            data[partner]["paths"] = new_segs_partner
            
            print(f"Merged main tracks {id_a} and {partner} at frame {frame}.")
            processed_pairs.add(pair)
    
    # Write the merged data to output3.json.
    with open("output3.json", "w") as f:
        json.dump(data, f, indent=4)
    
    print("Merging complete. Output written to output3.json")

if __name__ == "__main__":
    main()

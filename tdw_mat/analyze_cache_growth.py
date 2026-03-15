import os
import re
import numpy as np
from glob import glob

log_dir = "result_log/run_eval_60"
models = ["gpt-5-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"]
checkpoints = range(0, 6001, 500)

model_stats = {m: [] for m in models}

print(f"Analyzing logs in {log_dir}")

for model in models:
    # Find all files for this model
    # Pattern: *_{model}_plan_tracking.txt
    files = glob(os.path.join(log_dir, f"*_{model}_plan_tracking.txt"))
    print(f"Found {len(files)} files for {model}")
    
    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        episodes = []
        current_episode_events = []
        in_episode = False
        last_line_was_start = False
        
        for line in lines:
            if "[Frame 0] Plan started:" in line:
                if last_line_was_start:
                    continue # Ignore duplicate header
                
                # If we were already in an episode, finish it
                if in_episode:
                    episodes.append(current_episode_events)
                
                # Start new episode
                current_episode_events = [(0, 0)] # Initialize frame 0 with 0 cache
                in_episode = True
                last_line_was_start = True
            else:
                last_line_was_start = False
                if in_episode:
                    # [Frame 19] Cache lines: 0 -> 1 (change: +1)
                    match = re.search(r"\[Frame (\d+)\] Cache lines: \d+ -> (\d+)", line)
                    if match:
                        frame = int(match.group(1))
                        count = int(match.group(2))
                        current_episode_events.append((frame, count))
        
        # Append last episode
        if in_episode:
            episodes.append(current_episode_events)
            
        # Process episodes into sampling points
        for events in episodes:
            # Sort events by frame
            events.sort(key=lambda x: x[0])
            
            samples = []
            current_cache = 0
            evt_idx = 0
            
            for cp in checkpoints:
                # Update current_cache for all events happening up to this checkpoint
                while evt_idx < len(events) and events[evt_idx][0] <= cp:
                    current_cache = events[evt_idx][1]
                    evt_idx += 1
                samples.append(current_cache)
            
            model_stats[model].append(samples)
    
    print(f"  Processed {len(model_stats[model])} episodes for {model}")

print("\nResults (Average Cache Lines):")
# Print Header
header = "Frame" + "".join([f"\t{m}" for m in models])
print(header)

# Print Data
for i, cp in enumerate(checkpoints):
    row = f"{cp}"
    for model in models:
        data = model_stats[model]
        if data:
            avg_val = np.mean([ep[i] for ep in data])
            row += f"\t{avg_val:.2f}"
        else:
            row += f"\t0.00"
    print(row)

import os
import json
import re
from glob import glob
import numpy as np

base_path = "results/run_30"

def get_dir_stats(dir_path):
    stats = {
        "avg_latency": 0.0,
        "total_tokens": 0,
        "sr": 0.0
    }
    
    # 1. Latency (Time) & SR
    eval_result_path = os.path.join(dir_path, "run_eval", "eval_result.json")
    if os.path.exists(eval_result_path):
        try:
            with open(eval_result_path, 'r') as f:
                data = json.load(f)
                
                # SR (avg_finish)
                if "avg_finish" in data:
                    stats["sr"] = float(data["avg_finish"])
                
                # Latency
                # Prefer explicit average if available
                if "avg_episode_time" in data:
                    stats["avg_latency"] = float(data["avg_episode_time"])
                elif "episode_times" in data and data["episode_times"]:
                    times = [float(t) for t in data["episode_times"].values()]
                    stats["avg_latency"] = sum(times) / len(times)
        except Exception as e:
            print(f"Error reading {eval_result_path}: {e}")

    # 2. Tokens
    llm_path = os.path.join(dir_path, "run_eval", "LLM")
    if os.path.exists(llm_path):
        log_files = glob(os.path.join(llm_path, "chat_raw_v2_*.json"))
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    matches = re.findall(r"Total tokens:\s*(\d+)", content)
                    for m in matches:
                        stats["total_tokens"] += int(m)
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
                
    return stats

def main():
    if not os.path.exists(base_path):
        print(f"Path not found: {base_path}")
        return

    all_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    groups = {
        "2a (HE)": [],
        "5a": []
    }
    
    for d in all_dirs:
        # Check for 2a HE
        if "_2a_" in d and d.endswith("_HE"):
            groups["2a (HE)"].append(d)
        # Check for 5a (exclude HE just in case, though based on file names they shouldn't overlap in run_30)
        elif "_5a_" in d and not d.endswith("_HE"):
            groups["5a"].append(d)
            
    print(f"Found {len(groups['2a (HE)'])} 2a(HE) directories and {len(groups['5a'])} 5a directories.")
    
    print("-" * 100)
    print(f"{'Group':<10} | {'Count':<5} | {'Avg SR':<10} | {'Avg Latency (s)':<20} | {'Avg Total Tokens':<20}")
    print("-" * 100)
    
    for group_name, dirs in groups.items():
        latencies = []
        tokens = []
        srs = []
        
        for d in dirs:
            full_path = os.path.join(base_path, d)
            s = get_dir_stats(full_path)
            latencies.append(s["avg_latency"])
            tokens.append(s["total_tokens"])
            srs.append(s["sr"])
            
        if not latencies:
            print(f"{group_name:<10} | {0:<5} | {'N/A':<10} | {'N/A':<20} | {'N/A':<20}")
            continue
            
        avg_latency_group = sum(latencies) / len(latencies)
        avg_tokens_group = sum(tokens) / len(tokens)
        avg_sr_group = sum(srs) / len(srs)
        
        print(f"{group_name:<10} | {len(dirs):<5} | {avg_sr_group:<10.4f} | {avg_latency_group:<20.2f} | {avg_tokens_group:<20.2f}")

    print("-" * 100)
    print("\n[Detailed Breakdown]")
    for group_name, dirs in groups.items():
        print(f"\nGroup: {group_name}")
        for d in dirs:
            full_path = os.path.join(base_path, d)
            s = get_dir_stats(full_path)
            print(f"  - {d}: SR={s['sr']:.2f}, Latency={s['avg_latency']:.2f}s, Tokens={s['total_tokens']}")

if __name__ == "__main__":
    main()

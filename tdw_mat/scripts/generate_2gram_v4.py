#!/usr/bin/env python3
"""
Generate action_2gram_grouped_by_start_v4.txt from plan_sequence_agent_*.jsonl files.

v4 adds: known_targets_count, known_containers_count, unexplored_rooms
v4 also computes ACTION_WEIGHTS dynamically.

Usage:
    python scripts/generate_2gram_v4.py \
        --results_dir results/baseline/LMs-gpt-5-2025-08-07/run_1_test_2 \
        --episodes 1 2 3 4 \
        --output tdw-gym/action_2gram_grouped_by_start_v4.txt
"""

import argparse
import json
import os
import math
from collections import defaultdict


def load_plan_sequences(results_dir, episodes):
    """Load all plan_sequence JSONL files for given episodes."""
    all_steps = []
    for ep in episodes:
        ep_dir = os.path.join(results_dir, str(ep))
        for agent_id in [0, 1]:
            filepath = os.path.join(ep_dir, f"plan_sequence_agent_{agent_id}.jsonl")
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found, skipping")
                continue
            with open(filepath, 'r') as f:
                steps = []
                for line in f:
                    line = line.strip()
                    if line:
                        steps.append(json.loads(line))
                all_steps.append(steps)
    return all_steps


def compute_action_weights(all_sequences):
    """
    Compute ACTION_WEIGHTS as: # times action type selected / # times action type appeared in options.
    For denominator, each action type counts once per step regardless of how many options of that type exist.
    """
    selected_counts = defaultdict(int)
    appeared_counts = defaultdict(int)

    for steps in all_sequences:
        for step in steps:
            selected_type = step['selected_type']
            selected_counts[selected_type] += 1

            # Deduplicate available_types for this step
            unique_types = set(step['available_types'])
            for t in unique_types:
                appeared_counts[t] += 1

    weights = {}
    for action_type in appeared_counts:
        weights[action_type] = round(selected_counts.get(action_type, 0) / appeared_counts[action_type], 3)

    return weights


def normalize_type(selected_type):
    """Normalize action type to v2/v3 display name."""
    type_map = {
        'explore': 'explore',
        'goto': 'go to',
        'gograsp_target': 'go grasp target',
        'gograsp_container': 'go grasp container',
        'putin': 'put into container',
        'transport': 'transport',
        'sendmessage': 'send message',
    }
    return type_map.get(selected_type, selected_type)


def build_2grams(all_sequences):
    """
    Build 2-gram transitions from plan sequences.
    Returns dict: from_action -> list of transition records.
    Each transition record has:
        - next_action
        - action1_dur, action2_dur, action1_frame, action2_frame
        - rooms, held, container, satisfied
        - known_targets, known_containers, unexplored_rooms  (NEW in v4)
    """
    transitions = defaultdict(list)

    for steps in all_sequences:
        # Remove consecutive duplicates by selected_type
        deduped = []
        for step in steps:
            if not deduped or step['selected_type'] != deduped[-1]['selected_type']:
                deduped.append(step)

        for i in range(len(deduped) - 1):
            s1 = deduped[i]
            s2 = deduped[i + 1]

            from_action = s1['selected_type']
            to_action = s2['selected_type']

            action1_dur = s2['frame'] - s1['frame']
            action1_frame = s1['frame']
            action2_frame = s2['frame']

            # For action2_dur, use next transition's frame if available, else estimate
            if i + 2 < len(deduped):
                action2_dur = deduped[i + 2]['frame'] - s2['frame']
            else:
                action2_dur = action1_dur  # fallback estimate

            record = {
                'next_action': to_action,
                'action1_dur': action1_dur,
                'action2_dur': action2_dur,
                'action1_frame': action1_frame,
                'action2_frame': action2_frame,
                # State at time of action1 (from s1)
                'rooms': s1.get('rooms_explored_all', 0) + s1.get('rooms_explored_part', 0),
                'held': s1.get('held_count', 0),
                'container': s1.get('container_held', 0),
                'satisfied': s1.get('satisfied_count', 0),
                # NEW v4 fields
                'known_targets': s1.get('known_targets_count', 0),
                'known_containers': s1.get('known_containers_count', 0),
                'unexplored_rooms': s1.get('unexplored_rooms', 0),
            }
            transitions[from_action].append(record)

    return transitions


def stats(values):
    """Compute avg, med, std, min, max for a list of numbers."""
    if not values:
        return {'avg': 0, 'med': 0, 'std': 0.0, 'min': 0, 'max': 0}
    n = len(values)
    avg = sum(values) / n
    sorted_v = sorted(values)
    med = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    variance = sum((x - avg) ** 2 for x in values) / n if n > 1 else 0
    std = math.sqrt(variance)
    return {
        'avg': round(avg, 1) if avg != int(avg) else int(avg),
        'med': round(med, 1) if med != int(med) else int(med),
        'std': round(std, 1),
        'min': int(min(values)),
        'max': int(max(values)),
    }


def format_stat(name, s):
    """Format a stat dict as 'name:avg(med:X,std:Y,min:Z,max:W)'."""
    return f"{name}:{s['avg']}(med:{s['med']},std:{s['std']},min:{s['min']},max:{s['max']})"


def format_stat_no_std(name, s):
    """Format without std: 'name:avg(med:X,min:Z,max:W)' for duration fields."""
    return f"{name}:{s['avg']}(med:{s['med']},min:{s['min']},max:{s['max']})"


def generate_output(transitions, output_path):
    """Generate the v4 2-gram text file."""
    lines = []
    lines.append("=" * 80)
    lines.append("2-GRAM ACTION TRANSITIONS GROUPED BY STARTING ACTION")
    lines.append("(No Consecutive Duplicates, with Metadata) [v4: +known_targets, +known_containers, +unexplored_rooms]")
    lines.append("=" * 80)
    lines.append("")

    # Sort from_actions by total transition count (descending)
    sorted_actions = sorted(transitions.keys(), key=lambda a: len(transitions[a]), reverse=True)

    for from_action in sorted_actions:
        records = transitions[from_action]
        display_name = normalize_type(from_action)
        total = len(records)

        # Overall stats
        overall_a1_dur = stats([r['action1_dur'] for r in records])
        overall_a2_dur = stats([r['action2_dur'] for r in records])
        overall_a1_frame = stats([r['action1_frame'] for r in records])
        overall_a2_frame = stats([r['action2_frame'] for r in records])
        overall_rooms = stats([r['rooms'] for r in records])
        overall_held = stats([r['held'] for r in records])
        overall_container = stats([r['container'] for r in records])
        overall_satisfied = stats([r['satisfied'] for r in records])
        overall_known_targets = stats([r['known_targets'] for r in records])
        overall_known_containers = stats([r['known_containers'] for r in records])
        overall_unexplored = stats([r['unexplored_rooms'] for r in records])

        lines.append("")
        lines.append("=" * 80)
        lines.append(f"FROM: {display_name}")
        lines.append(f"Total transitions: {total}")
        lines.append(
            f"Overall: "
            f"[{format_stat_no_std('action1_dur', overall_a1_dur)}, {format_stat_no_std('action2_dur', overall_a2_dur)}] "
            f"[{format_stat('action1@frame', overall_a1_frame)}, {format_stat('action2@frame', overall_a2_frame)}] "
            f"[{format_stat_no_std('rooms', overall_rooms)}, {format_stat_no_std('held', overall_held)}, "
            f"{format_stat_no_std('container', overall_container)}, {format_stat_no_std('satisfied', overall_satisfied)}, "
            f"{format_stat_no_std('known_targets', overall_known_targets)}, {format_stat_no_std('known_containers', overall_known_containers)}, "
            f"{format_stat_no_std('unexplored_rooms', overall_unexplored)}]"
        )
        lines.append("=" * 80)

        # Group by next_action
        by_next = defaultdict(list)
        for r in records:
            by_next[r['next_action']].append(r)

        # Sort by count descending
        sorted_next = sorted(by_next.items(), key=lambda x: len(x[1]), reverse=True)

        for next_action, next_records in sorted_next:
            count = len(next_records)
            pct = count / total * 100
            next_display = normalize_type(next_action)

            a1_dur = stats([r['action1_dur'] for r in next_records])
            a2_dur = stats([r['action2_dur'] for r in next_records])
            a1_frame = stats([r['action1_frame'] for r in next_records])
            a2_frame = stats([r['action2_frame'] for r in next_records])
            rooms_s = stats([r['rooms'] for r in next_records])
            held_s = stats([r['held'] for r in next_records])
            container_s = stats([r['container'] for r in next_records])
            satisfied_s = stats([r['satisfied'] for r in next_records])
            known_targets_s = stats([r['known_targets'] for r in next_records])
            known_containers_s = stats([r['known_containers'] for r in next_records])
            unexplored_s = stats([r['unexplored_rooms'] for r in next_records])

            lines.append("")
            lines.append(f"  → {next_display:<30} [{count:>4}] ({pct:>5.1f}%)")
            lines.append(
                f"    [{format_stat_no_std('action1_dur', a1_dur)}, {format_stat_no_std('action2_dur', a2_dur)}] "
                f"[{format_stat('action1@frame', a1_frame)}, {format_stat('action2@frame', a2_frame)}] "
                f"[{format_stat_no_std('rooms', rooms_s)}, {format_stat_no_std('held', held_s)}, "
                f"{format_stat_no_std('container', container_s)}, {format_stat_no_std('satisfied', satisfied_s)}, "
                f"{format_stat_no_std('known_targets', known_targets_s)}, {format_stat_no_std('known_containers', known_containers_s)}, "
                f"{format_stat_no_std('unexplored_rooms', unexplored_s)}]"
            )

        lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 2-gram v4 action transition file")
    parser.add_argument('--results_dir', required=True, help="Path to run results directory")
    parser.add_argument('--episodes', nargs='+', type=int, required=True, help="Episode numbers to include")
    parser.add_argument('--output', required=True, help="Output file path")
    args = parser.parse_args()

    print(f"Loading episodes {args.episodes} from {args.results_dir}")
    all_sequences = load_plan_sequences(args.results_dir, args.episodes)
    print(f"Loaded {len(all_sequences)} agent sequences")

    # Compute action weights
    weights = compute_action_weights(all_sequences)
    print("\n=== ACTION_WEIGHTS (selected/appeared) ===")
    for action, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action}: {w}")

    # Build 2-grams
    transitions = build_2grams(all_sequences)
    total_transitions = sum(len(v) for v in transitions.values())
    print(f"\nTotal 2-gram transitions: {total_transitions}")

    # Generate output
    generate_output(transitions, args.output)

    # Also output weights as Python dict for easy copy-paste
    print("\n=== Copy-paste for action_cache.py ===")
    print("ACTION_WEIGHTS = {")
    for action, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"    '{action}': {w},")
    print("}")


if __name__ == '__main__':
    main()

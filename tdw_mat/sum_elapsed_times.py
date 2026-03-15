import json

# JSON 파일 경로
json_file = "results/LMs-gpt-5/run_parallel_gpt-5/eval_result.json"

# JSON 파일 읽기
with open(json_file, 'r') as f:
    data = json.load(f)

# elapsed_times 합계 계산
total_elapsed_time = 0
episode_count = 0

for episode_id, episode_data in data['episode_results'].items():
    elapsed_time = episode_data['elapsed_times']
    total_elapsed_time += elapsed_time
    episode_count += 1
    print(f"Episode {episode_id}: {elapsed_time:.2f} seconds")

print("\n" + "="*50)
print(f"총 에피소드 수: {episode_count}")
print(f"총 elapsed time: {total_elapsed_time:.2f} seconds")
print(f"총 elapsed time: {total_elapsed_time/60:.2f} minutes")
print(f"총 elapsed time: {total_elapsed_time/3600:.2f} hours")
print(f"평균 elapsed time: {total_elapsed_time/episode_count:.2f} seconds per episode")

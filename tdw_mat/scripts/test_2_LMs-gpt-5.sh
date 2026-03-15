data_prefix=dataset/test_2/
split="test_2"

branch_name=$(git rev-parse --abbrev-ref HEAD)

lm_id=gpt-5-2025-08-07
port=1071
pkill -f -9 "port $port"

DISPLAY=:1 python3 tdw-gym/challenge.py \
--output_dir results/${branch_name} \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1_${split} \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.txt \
--max_tokens 256 \
--cot \
--data_prefix $data_prefix \
--eval_episodes 0 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
--screen_size 256 \
--no_save_img

pkill -f -9 "port $port"

lm_id=gpt-5-mini-2025-08-07
port=1071
pkill -f -9 "port $port"

DISPLAY=:1 python3 tdw-gym/challenge.py \
--output_dir results/${branch_name} \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1_${split} \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.txt \
--max_tokens 256 \
--cot \
--data_prefix $data_prefix \
--eval_episodes 0 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
--screen_size 256 \
--no_save_img

pkill -f -9 "port $port"

lm_id=gpt-5-nano-2025-08-07
port=1071
pkill -f -9 "port $port"

DISPLAY=:1 python3 tdw-gym/challenge.py \
--output_dir results/${branch_name} \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1_${split} \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.txt \
--max_tokens 256 \
--cot \
--data_prefix $data_prefix \
--eval_episodes 0 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
--screen_size 256 \
--no_save_img

pkill -f -9 "port $port"

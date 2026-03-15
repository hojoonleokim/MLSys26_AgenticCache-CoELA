#!/bin/bash

# Configuration
PORT=1071
MAX_FRAMES=3000
SCREEN_SIZE=256
EVAL_EPISODES="-1"  # Evaluate all episodes in the dataset

# Function to run evaluation
run_evaluation() {
    DATA_PREFIX=$1
    EXPERIMENT_NAME=$2
    LM_ID=$3
    
    # Check if results already exist
    RESULT_DIR="results/${EXPERIMENT_NAME}/run_eval"
    if [ -d "$RESULT_DIR" ]; then
        echo "⏭️  Skipping $EXPERIMENT_NAME with $LM_ID (results already exist)"
        return 0
    fi
    
    echo "Starting evaluation for $EXPERIMENT_NAME with model $LM_ID..."
    echo "Data prefix: $DATA_PREFIX"
    
    # Kill any existing TDW processes on the port
    pkill -f -9 "port $PORT"
    sleep 2

    # Run challenge
    DISPLAY=:1 python3 tdw-gym/challenge.py \
        --output_dir results \
        --lm_id $LM_ID \
        --experiment_name $EXPERIMENT_NAME \
        --run_id run_eval \
        --port $PORT \
        --agents lm_agent lm_agent \
        --communication \
        --prompt_template_path LLM/prompt_com.txt \
        --max_tokens 256 \
        --data_prefix $DATA_PREFIX \
        --data_path test_env.json \
        --eval_episodes $EVAL_EPISODES \
        --screen_size $SCREEN_SIZE \
        --max_frames $MAX_FRAMES \
        --no_save_img \
        --cot \
        --no_cache_loading

    echo "Evaluation for $EXPERIMENT_NAME with $LM_ID completed."
    sleep 5
}

# List of models to evaluate
MODELS=("gpt-5-2025-08-07" "gpt-5-mini-2025-08-07" "gpt-5-nano-2025-08-07")

# List of datasets
DATASETS=(
    "dataset/large_map_10_objs_4a_food/:eval_10objs_4a_food"
    "dataset/large_map_10_objs_4a_stuff/:eval_10objs_4a_stuff"
    "dataset/large_map_10_objs_2a_food/:eval_10objs_2a_food"
    "dataset/large_map_10_objs_2a_stuff/:eval_10objs_2a_stuff"
    "dataset/large_map_10_objs_5a_food/:eval_10objs_5a_food"
    "dataset/large_map_10_objs_5a_stuff/:eval_10objs_5a_stuff"
)

# Run evaluations for all models and datasets
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Starting evaluations with model: $MODEL"
    echo "=========================================="
    
    for DATASET in "${DATASETS[@]}"; do
        DATA_PREFIX="${DATASET%%:*}"
        EXPERIMENT_NAME="${DATASET##*:}_${MODEL}"
        
        run_evaluation "$DATA_PREFIX" "$EXPERIMENT_NAME" "$MODEL"
    done
    
    echo "Completed all datasets for model: $MODEL"
    echo ""
done

echo "All evaluations completed for all models!"

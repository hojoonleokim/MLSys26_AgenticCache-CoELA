#!/bin/bash

# Function to generate dataset
generate_dataset() {
    SCENE=$1
    LAYOUT=$2
    ID=$3
    DATASET_NAME=$4
    TARGET_COUNT=$5
    TASK_MODE=$6

    # Check if dataset already exists
    if [ -d "dataset/$DATASET_NAME" ]; then
        echo "⏭️  Skipping $DATASET_NAME (already exists)"
        return 0
    fi

    echo "Generating dataset: $DATASET_NAME with Scene: $SCENE, Objects: $TARGET_COUNT, Task: $TASK_MODE"

    # 1. Create dataset directory
    mkdir -p dataset/$DATASET_NAME

    # 2. Run scene generator
    export PYTHONPATH=$PYTHONPATH:$(pwd)/scene_generator
    DISPLAY=:1 python3 scene_generator/scene_generate.py $SCENE $LAYOUT $ID $DATASET_NAME $TARGET_COUNT $TASK_MODE

    # 3. Copy necessary config files
    cp dataset/list.json dataset/$DATASET_NAME/list.json
    cp dataset/name_map.json dataset/$DATASET_NAME/name_map.json
    cp dataset/room_types.json dataset/$DATASET_NAME/room_types.json
    cp dataset/object_scale.json dataset/$DATASET_NAME/object_scale.json

    # 4. Create test_env.json
    cat <<EOF > dataset/$DATASET_NAME/test_env.json
[
    {
        "scene": "$SCENE",
        "layout": "$LAYOUT",
        "task": "$TASK_MODE",
        "seed": 1234
    }
]
EOF
    echo "Dataset generated at dataset/$DATASET_NAME"
}

# Generate for Scene 4a (Large Map)
generate_dataset "4a" "0" "0" "large_map_30_objs_4a_food" "30" "food"
generate_dataset "4a" "0" "0" "large_map_30_objs_4a_stuff" "30" "stuff"
generate_dataset "4a" "0" "0" "large_map_10_objs_4a_food" "10" "food"
generate_dataset "4a" "0" "0" "large_map_10_objs_4a_stuff" "10" "stuff"

# Generate for Scene 2a (Another Large Map)
generate_dataset "2a" "0" "2" "large_map_30_objs_2a_food" "30" "food"
generate_dataset "2a" "0" "2" "large_map_30_objs_2a_stuff" "30" "stuff"
generate_dataset "2a" "0" "2" "large_map_10_objs_2a_food" "10" "food"
generate_dataset "2a" "0" "2" "large_map_10_objs_2a_stuff" "10" "stuff"

# Generate for Scene 5a (Large Map)
generate_dataset "5a" "0" "0" "large_map_30_objs_5a_food" "30" "food"
generate_dataset "5a" "0" "0" "large_map_30_objs_5a_stuff" "30" "stuff"
generate_dataset "5a" "0" "0" "large_map_10_objs_5a_food" "10" "food"
generate_dataset "5a" "0" "0" "large_map_10_objs_5a_stuff" "10" "stuff"

echo "All datasets generated."
echo "Run examples (30 objects):"
echo "python3 tdw-gym/challenge.py --data_prefix dataset/large_map_30_objs_4a_food/ --data_path test_env.json --agents lm_agent --port 1071"
echo "python3 tdw-gym/challenge.py --data_prefix dataset/large_map_30_objs_4a_stuff/ --data_path test_env.json --agents lm_agent --port 1071"
echo ""
echo "Run examples (10 objects):"
echo "python3 tdw-gym/challenge.py --data_prefix dataset/large_map_10_objs_4a_food/ --data_path test_env.json --agents lm_agent --port 1071"
echo "python3 tdw-gym/challenge.py --data_prefix dataset/large_map_10_objs_4a_stuff/ --data_path test_env.json --agents lm_agent --port 1071"

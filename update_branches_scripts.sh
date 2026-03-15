#!/bin/bash
set -e

# Store current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Get reference to the files we need to ensure exist across all branches
mkdir -p /tmp/coela_scripts
cp tdw_mat/scripts/test_LMs-gpt-5.sh /tmp/coela_scripts/test_LMs-gpt-5.sh
cp tdw_mat/scripts/train_LMs-gpt-5.sh /tmp/coela_scripts/train_LMs-gpt-5.sh

# Now process each branch
for branch in parallel speculative; do
    echo "Processing branch: $branch"
    
    # Checkout the branch
    git checkout $branch
    git pull origin $branch || true
    
    # Remove all other scripts in tdw_mat/scripts/ safely
    # For branches where files might be in .gitignore or not tracked
    find tdw_mat/scripts/ -maxdepth 1 -type f ! -name "train_LMs-gpt-5.sh" ! -name "test_LMs-gpt-5.sh" -exec rm -f {} + || true
    rm -rf tdw_mat/scripts/wo_gt_mask || true
    
    # Track files removed locally in git
    git add -u tdw_mat/scripts/ || true
    
    # 1. Update test_LMs-gpt-5.sh and train_LMs-gpt-5.sh
    mkdir -p tdw_mat/scripts/
    cp /tmp/coela_scripts/test_LMs-gpt-5.sh tdw_mat/scripts/test_LMs-gpt-5.sh
    cp /tmp/coela_scripts/train_LMs-gpt-5.sh tdw_mat/scripts/train_LMs-gpt-5.sh
    chmod +x tdw_mat/scripts/test_LMs-gpt-5.sh
    chmod +x tdw_mat/scripts/train_LMs-gpt-5.sh
    
    # Add new scripts to tracking (using -f in case of .gitignore)
    git add -f tdw_mat/scripts/test_LMs-gpt-5.sh tdw_mat/scripts/train_LMs-gpt-5.sh
    
    # 2. Rename directories if they exist (using git mv to properly track renames)
    if [ -d "tdw_mat/dataset/test_1" ]; then
        git mv tdw_mat/dataset/test_1 tdw_mat/dataset/test_1 || true
    fi
    if [ -d "tdw_mat/dataset/test_2" ]; then
        git mv tdw_mat/dataset/test_2 tdw_mat/dataset/test_2 || true
    fi
    
    # 3. Find and replace all occurrences of test_1 -> test_1 and test_2 -> test_2
    # Only process python scripts, shell scripts, txt files and csvs
    find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.txt" -o -name "*.csv" -o -name "*.md" \) -exec sed -i 's/test_1/test_1/g' {} +
    find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.txt" -o -name "*.csv" -o -name "*.md" \) -exec sed -i 's/test_2/test_2/g' {} +
    
    # Stage all changes
    git add -A
    
    # Commit if there are changes
    if ! git diff --cached --quiet; then
        git commit -m "Refactor: rename dataset dirs, clean up scripts, add gpt-5 scripts"
        git push origin $branch
    else
        echo "No changes needed for $branch"
    fi
done

# Go back to original branch (we started in parallel)
git checkout baseline
echo "Done!"

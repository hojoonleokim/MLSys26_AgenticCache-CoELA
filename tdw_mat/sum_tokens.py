import re
import os
from collections import defaultdict

def sum_tokens_by_model(directory):
    """Sum tokens by model across all agents in the directory"""
    
    # Dictionary to store tokens by model
    model_tokens = defaultdict(lambda: {'input': 0, 'output': 0, 'total': 0})
    
    llm_dir = os.path.join(directory, 'LLM')
    
    if not os.path.exists(llm_dir):
        print(f"Error: Directory {llm_dir} does not exist")
        return
    
    # Find all chat_raw files
    files = [f for f in os.listdir(llm_dir) if f.startswith('chat_raw') and f.endswith('.json')]
    
    if not files:
        print(f"No chat_raw files found in {llm_dir}")
        return
    
    print(f"Found {len(files)} files in {llm_dir}\n")
    
    for filename in files:
        # Extract model name from filename
        # Format: chat_raw_v2_{model}_agent_{n}.json
        match = re.search(r'chat_raw_v2_(.+?)_agent_\d+\.json', filename)
        if not match:
            print(f"Skipping {filename} - couldn't extract model name")
            continue
            
        model_name = match.group(1)
        filepath = os.path.join(llm_dir, filename)
        
        print(f"Processing: {filename} (model: {model_name})")
        
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract all token counts using regex
            input_tokens = re.findall(r'Input tokens: (\d+)', content)
            output_tokens = re.findall(r'Output tokens: (\d+)', content)
            total_tokens_list = re.findall(r'Total tokens: (\d+)', content)
            
            file_input = sum(int(x) for x in input_tokens)
            file_output = sum(int(x) for x in output_tokens)
            file_total = sum(int(x) for x in total_tokens_list)
            
            print(f"  - Input tokens: {file_input:,}")
            print(f"  - Output tokens: {file_output:,}")
            print(f"  - Total tokens: {file_total:,}\n")
            
            # Add to model totals
            model_tokens[model_name]['input'] += file_input
            model_tokens[model_name]['output'] += file_output
            model_tokens[model_name]['total'] += file_total
    
    # Print summary by model
    print("\n" + "="*60)
    print("TOKEN SUMMARY BY MODEL (all agents combined):")
    print("="*60)
    
    for model_name in sorted(model_tokens.keys()):
        tokens = model_tokens[model_name]
        print(f"\nModel: {model_name}")
        print(f"  Input tokens:  {tokens['input']:,}")
        print(f"  Output tokens: {tokens['output']:,}")
        print(f"  Total tokens:  {tokens['total']:,}")
    
    # Print grand total
    grand_input = sum(t['input'] for t in model_tokens.values())
    grand_output = sum(t['output'] for t in model_tokens.values())
    grand_total = sum(t['total'] for t in model_tokens.values())
    
    print("\n" + "="*60)
    print("GRAND TOTAL (all models, all agents):")
    print(f"  Input tokens:  {grand_input:,}")
    print(f"  Output tokens: {grand_output:,}")
    print(f"  Total tokens:  {grand_total:,}")
    print("="*60)
    
    return model_tokens


if __name__ == "__main__":
    directory = "results/run_parallel_gpt-5"
    sum_tokens_by_model(directory)

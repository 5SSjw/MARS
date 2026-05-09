import json
import os
import glob
import argparse
from transformers import AutoTokenizer
import numpy as np

"""
python /work/xinyu/MARS/mars/evaluation/speed_batch.py --folder /work/xinyu/MARS/mars/ablation3/qwen3-32b --prefix humaneval --tokenizer /work/xinyu/models/Qwen3-32B
"""

def calculate_speed(data, tokenizer=None, use_new_tokens=True):
    """Calculate speed from data entries."""
    speeds = []
    total_time = 0
    total_token = 0
    
    for datapoint in data:
        answer = datapoint["choices"][0]['turns']
        if use_new_tokens:
            tokens = sum(datapoint["choices"][0]['new_tokens'])
        else:
            tokens = 0
            for i in answer:
                tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds.append(tokens / times)
        total_time += times
        total_token += tokens
    
    return speeds, total_time, total_token

def calculate_acceptance_length(file_path):
    """Calculate average acceptance length."""
    new_tokens_sum = 0
    idxs_sum = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if "choices" in data:
                for choice in data["choices"]:
                    new_tokens_sum += sum(choice.get("new_tokens", []))
                    idxs_sum += sum(choice.get("idxs", []))
    
    result = new_tokens_sum / idxs_sum if idxs_sum != 0 else None
    return new_tokens_sum, idxs_sum, result

def process_file_pair(jsonl_file, jsonl_file_base, tokenizer):
    """Process a pair of experiment and baseline files."""
    # Load experiment data
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    
    # Calculate experiment speed
    speeds, _, _ = calculate_speed(data, use_new_tokens=True)
    
    # Load baseline data
    data_base = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data_base.append(json_obj)
    
    # Calculate baseline speed
    speeds0, total_time, total_token = calculate_speed(data_base, tokenizer, use_new_tokens=False)
    
    # Calculate metrics
    speed_exp = np.array(speeds).mean()
    speed_base = np.array(speeds0).mean()
    speedup_ratio = speed_exp / speed_base
    
    # Calculate acceptance length
    new_tokens_sum, idxs_sum, avg_acc_len = calculate_acceptance_length(jsonl_file)
    
    return {
        'speed_exp': speed_exp,
        'speed_base': speed_base,
        'speedup_ratio': speedup_ratio,
        'new_tokens_sum': new_tokens_sum,
        'idxs_sum': idxs_sum,
        'avg_acc_len': avg_acc_len
    }

def main():
    parser = argparse.ArgumentParser(description='Batch speed comparison for MARS evaluation')
    parser.add_argument('--folder', type=str, required=True, 
                        help='Path to folder containing jsonl files')
    parser.add_argument('--prefix', type=str, required=True,
                        help='Prefix of files to compare (e.g., gsm8k)')
    parser.add_argument('--tokenizer', type=str, default="/workspace/models/Qwen3-8B",
                        help='Path to tokenizer')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Find baseline file
    baseline_pattern = os.path.join(args.folder, f"{args.prefix}*_baseline*.jsonl")
    baseline_files = glob.glob(baseline_pattern)
    
    if len(baseline_files) == 0:
        print(f"Error: No baseline file found matching pattern: {baseline_pattern}")
        return
    elif len(baseline_files) > 1:
        print(f"Warning: Multiple baseline files found, using first one: {baseline_files[0]}")
    
    baseline_file = baseline_files[0]
    print(f"Baseline file: {os.path.basename(baseline_file)}")
    print("=" * 80)
    
    # Find all experiment files (exclude baseline)
    all_pattern = os.path.join(args.folder, f"{args.prefix}*.jsonl")
    all_files = glob.glob(all_pattern)
    
    # Filter out baseline files
    exp_files = [f for f in all_files if '_baseline' not in os.path.basename(f)]
    
    if len(exp_files) == 0:
        print(f"Error: No experiment files found matching pattern: {all_pattern}")
        return
    
    # Sort files for consistent output
    exp_files.sort()
    
    # Process each experiment file
    results = []
    for exp_file in exp_files:
        print(f"\nProcessing: {os.path.basename(exp_file)}")
        print("-" * 60)
        
        try:
            metrics = process_file_pair(exp_file, baseline_file, tokenizer)
            results.append({
                'file': os.path.basename(exp_file),
                **metrics
            })
            
            print(f"  Speed (exp):     {metrics['speed_exp']:.2f} tokens/s")
            print(f"  Speed (base):    {metrics['speed_base']:.2f} tokens/s")
            print(f"  Speedup Ratio:   {metrics['speedup_ratio']:.4f}x")
            print(f"  Avg Acc Length:  {metrics['avg_acc_len']:.4f}")
            print(f"  (New Tokens: {metrics['new_tokens_sum']}, Idxs: {metrics['idxs_sum']})")
        except Exception as e:
            print(f"  Error processing file: {e}")
    
    # Summary table
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'File':<50} {'Speedup':>10} {'Avg Acc Len':>12}")
        print("-" * 80)
        for r in results:
            print(f"{r['file']:<50} {r['speedup_ratio']:>10.4f}x {r['avg_acc_len']:>12.4f}")

if __name__ == "__main__":
    main()

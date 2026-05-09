import json
from transformers import AutoTokenizer
import numpy as np
import os
from pathlib import Path
from collections import defaultdict

# ============== 配置区域 ==============
# TOKENIZER_PATH = "/workspace/models/Llama-3.1-8B-Instruct"
TOKENIZER_PATH = "/work/xinyu/models/Llama-3.3-70B-Instruct"
# TOKENIZER_PATH = "/work/xinyu/models/vicuna-13b-v1.3"
# TOKENIZER_PATH = "/work/xinyu/models/DeepSeek-R1-Distill-Llama-8B"
# TOKENIZER_PATH = "/work/xinyu/models/Qwen3-32B"


# 三个文件夹路径 (分别存放三种结果)
FOLDER_RATIO = "/work/xinyu/MARS/mars/final/llama3-70b"
FOLDER_MARS = "/work/xinyu/MARS1/mars/final/llama3-70b"
FOLDER_BASELINE = "/work/xinyu/MARS1/mars/final/llama3-70b"

# 实验后缀 (文件名中的后缀)
SUFFIX_BASELINE = "baseline"
SUFFIX_RATIO = "ratio0.9"
SUFFIX_MARS = "mars"

# Benchmark短名映射 (用于表格显示)
BENCH_SHORT_NAMES = {
    "mt_bench": "MT-bench",
    "humaneval": "Humaneval",
    "gsm8k": "GSM8K",
    "alpaca": "Alpaca",
    "sum": "CNN/DM",
}

# ======================================

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data


def calc_speed_mars(data):
    """计算MARS/RATIO实验的速度 (使用new_tokens)"""
    speeds = []
    for datapoint in data:
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        if times > 0:
            speeds.append(tokens / times)
    return speeds


def calc_speed_baseline(data, tokenizer):
    """计算baseline的速度 (需要tokenizer重新计算tokens)"""
    speeds = []
    total_time = 0
    total_token = 0
    for datapoint in data:
        answer = datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        if times > 0:
            speeds.append(tokens / times)
        total_time += times
        total_token += tokens
    return speeds, total_time, total_token


def calc_avg_accept_length(data):
    """计算平均接受长度 (τ)"""
    new_tokens_sum = 0
    idxs_sum = 0
    for datapoint in data:
        if "choices" in datapoint:
            for choice in datapoint["choices"]:
                new_tokens_sum += sum(choice.get("new_tokens", []))
                idxs_sum += sum(choice.get("idxs", []))
    result = new_tokens_sum / idxs_sum if idxs_sum != 0 else None
    return new_tokens_sum, idxs_sum, result


def find_benchmarks(folder, suffix):
    """查找文件夹中所有指定后缀的benchmark"""
    benchmarks = set()
    folder_path = Path(folder)
    if not folder_path.exists():
        return benchmarks
    for f in folder_path.glob(f"*_{suffix}.jsonl"):
        # 文件名格式: benchmark_t6_t1.0_suffix.jsonl
        # 提取 benchmark 名称 (第一个下划线之前的部分)
        basename = f.stem.replace(f"_{suffix}", "")
        # 提取benchmark短名 (如 mt_bench, humaneval 等)
        bench_name = basename.split("_t")[0]
        benchmarks.add((bench_name, basename))
    return benchmarks


def get_short_name(bench_name):
    """获取benchmark的短名用于显示"""
    return BENCH_SHORT_NAMES.get(bench_name, bench_name)


def main():
    print("=" * 80)
    print("实验对比分析")
    print("=" * 80)
    
    # 找到所有benchmark (以baseline为准)
    benchmarks = find_benchmarks(FOLDER_BASELINE, SUFFIX_BASELINE)
    
    if not benchmarks:
        print(f"错误: 在 {FOLDER_BASELINE} 中未找到任何 *_{SUFFIX_BASELINE}.jsonl 文件")
        return
    
    # 按benchmark名称排序
    benchmarks = sorted(benchmarks, key=lambda x: x[0])
    print(f"\n找到 {len(benchmarks)} 个benchmark")
    
    # 存储结果
    results_mars = {}
    results_ratio = {}
    
    for bench_name, basename in benchmarks:
        # 构建文件路径
        file_baseline = os.path.join(FOLDER_BASELINE, f"{basename}_{SUFFIX_BASELINE}.jsonl")
        file_ratio = os.path.join(FOLDER_RATIO, f"{basename}_{SUFFIX_RATIO}.jsonl")
        file_mars = os.path.join(FOLDER_MARS, f"{basename}_{SUFFIX_MARS}.jsonl")
        
        # 加载baseline数据
        if not os.path.exists(file_baseline):
            print(f"  [跳过] baseline文件不存在: {file_baseline}")
            continue
        
        data_baseline = load_jsonl(file_baseline)
        speeds_baseline, _, _ = calc_speed_baseline(data_baseline, tokenizer)
        speed_baseline_mean = np.array(speeds_baseline).mean()
        
        # 处理 mars vs baseline
        if os.path.exists(file_mars):
            data_mars = load_jsonl(file_mars)
            speeds_mars = calc_speed_mars(data_mars)
            speed_mars_mean = np.array(speeds_mars).mean()
            mars_speedup = speed_mars_mean / speed_baseline_mean
            _, _, avg_acc_len = calc_avg_accept_length(data_mars)
            results_mars[bench_name] = {
                'speedup': mars_speedup,
                'tau': avg_acc_len
            }
        
        # 处理 ratio0.9 vs baseline
        if os.path.exists(file_ratio):
            data_ratio = load_jsonl(file_ratio)
            speeds_ratio = calc_speed_mars(data_ratio)
            speed_ratio_mean = np.array(speeds_ratio).mean()
            ratio_speedup = speed_ratio_mean / speed_baseline_mean
            _, _, avg_acc_len = calc_avg_accept_length(data_ratio)
            results_ratio[bench_name] = {
                'speedup': ratio_speedup,
                'tau': avg_acc_len
            }
    
    # 获取所有benchmark名称
    all_benchmarks = sorted(set(results_mars.keys()) | set(results_ratio.keys()))
    
    # 计算Mean
    mars_speedups = [results_mars[b]['speedup'] for b in all_benchmarks if b in results_mars]
    mars_taus = [results_mars[b]['tau'] for b in all_benchmarks if b in results_mars and results_mars[b]['tau']]
    ratio_speedups = [results_ratio[b]['speedup'] for b in all_benchmarks if b in results_ratio]
    ratio_taus = [results_ratio[b]['tau'] for b in all_benchmarks if b in results_ratio and results_ratio[b]['tau']]
    
    # 打印表格 (论文格式)
    print("\n")
    print("=" * 120)
    print("论文格式表格")
    print("=" * 120)
    
    # 表头第一行
    header1_parts = [""]
    for bench in all_benchmarks:
        header1_parts.append(get_short_name(bench))
    header1_parts.append("Mean")
    
    # 计算每列宽度
    col_width = 16
    
    # 打印第一行表头 (benchmark名称)
    print("\t" + "\t\t".join([get_short_name(b) for b in all_benchmarks]) + "\t\tMean")
    
    # 打印第二行表头 (Speedup 和 τ)
    header2 = "Method\t"
    for _ in all_benchmarks:
        header2 += "Speedup\tτ\t"
    header2 += "Speedup\tτ"
    print(header2)
    
    # 打印 Mars 行
    mars_row = "Mars\t"
    for bench in all_benchmarks:
        if bench in results_mars:
            mars_row += f"{results_mars[bench]['speedup']:.2f}\t{results_mars[bench]['tau']:.2f}\t"
        else:
            mars_row += "N/A\tN/A\t"
    # Mean
    mars_mean_speedup = np.mean(mars_speedups) if mars_speedups else 0
    mars_mean_tau = np.mean(mars_taus) if mars_taus else 0
    mars_row += f"{mars_mean_speedup:.3f}\t{mars_mean_tau:.2f}"
    print(mars_row)
    
    # 打印 ours (ratio0.9) 行
    ratio_row = "ours\t"
    for bench in all_benchmarks:
        if bench in results_ratio:
            ratio_row += f"{results_ratio[bench]['speedup']:.2f}\t{results_ratio[bench]['tau']:.2f}\t"
        else:
            ratio_row += "N/A\tN/A\t"
    # Mean
    ratio_mean_speedup = np.mean(ratio_speedups) if ratio_speedups else 0
    ratio_mean_tau = np.mean(ratio_taus) if ratio_taus else 0
    ratio_row += f"{ratio_mean_speedup:.3f}\t{ratio_mean_tau:.2f}"
    print(ratio_row)
    
    # 打印可直接复制到Excel/LaTeX的格式
    print("\n")
    print("=" * 120)
    print("可复制格式 (Tab分隔，可直接粘贴到Excel)")
    print("=" * 120)
    
    # 表头
    header = "\t"
    for bench in all_benchmarks:
        header += f"{get_short_name(bench)}\t\t"
    header += "Mean\t"
    print(header)
    
    subheader = "Method\t"
    for _ in all_benchmarks:
        subheader += "Speedup\tτ\t"
    subheader += "Speedup\tτ"
    print(subheader)
    
    # Mars
    print(mars_row)
    # ours
    print(ratio_row)
    
    # 打印总结
    print("\n")
    print("=" * 80)
    print("实验总结")
    print("=" * 80)
    print(f"\n[Mars]")
    print(f"  - 平均加速倍率 (Mean Speedup): {mars_mean_speedup:.3f}x")
    print(f"  - 平均接受长度 (Mean τ): {mars_mean_tau:.2f}")
    print(f"\n[ours (ratio0.9)]")
    print(f"  - 平均加速倍率 (Mean Speedup): {ratio_mean_speedup:.3f}x")
    print(f"  - 平均接受长度 (Mean τ): {ratio_mean_tau:.2f}")


if __name__ == "__main__":
    main()
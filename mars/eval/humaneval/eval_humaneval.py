#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate HumanEval accuracy from already generated predictions (your JSONL format).
Only evaluates the tasks you actually attempted.

Usage (batch mode - recommended):
python /work/xinyu/MARS/mars/eval/humaneval/eval_humaneval.py \
  --input_dir /work/xinyu/MARS/fast-llm-inference/speculative_decoding/results/llama3-8b-70b \
  --save_dir  /work/xinyu/MARS/mars/output/humaneaval \
  --timeout 7 --n-workers 8

Usage (single file mode - legacy):
python /work/xinyu/MARS/mars/eval/humaneval/eval_humaneval.py \
  --pred_path /path/to/humaneval_xxx.jsonl \
  --save_dir  /path/to/output \
  --timeout 7 --n-workers 8
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, Any, List, Tuple

PROBLEM_FILE = "/work/xinyu/MARS/mars/eval/humaneval/data/HumanEval.jsonl.gz"

def add_project_root_to_syspath(project_root: str):
    if project_root and os.path.isdir(project_root) and project_root not in sys.path:
        sys.path.append(project_root)

def load_all_problems(problem_file: str) -> Dict[str, Any]:
    """Load FULL HumanEval problems using utils.data.read_problems."""
    from utils.data import read_problems
    if not os.path.exists(problem_file):
        raise FileNotFoundError(f"Cannot find HumanEval at: {problem_file}")
    return read_problems(problem_file)  # dict: {task_id: {...}}

def build_qid2task_map(problems: Dict[str, Any]) -> Dict[int, str]:
    """Map numeric id (e.g., 78) -> task_id (e.g., HumanEval/78)."""
    mapping = {}
    for task_id in problems.keys():
        tail = task_id.split("/")[-1]
        try:
            idx = int(tail)
            mapping[idx] = task_id
        except ValueError:
            pass
    if not mapping:
        raise RuntimeError("Failed to build question_id -> task_id mapping from problems.")
    return mapping

_CODE_BLOCK_PATTERNS = [
    r"```python(.*?)```",
    r"```py(.*?)```",
    r"```(.*?)```",
]

def extract_code_from_text(text: str) -> str:
    # python-tagged blocks (prefer last with 'def ')
    for pat in _CODE_BLOCK_PATTERNS[:2]:
        blocks = re.findall(pat, text, re.DOTALL | re.IGNORECASE)
        if blocks:
            for blk in reversed(blocks):
                if "def " in blk:
                    return blk.strip()
            return blocks[-1].strip()
    # any fenced blocks
    blocks = re.findall(_CODE_BLOCK_PATTERNS[2], text, re.DOTALL)
    if blocks:
        for blk in reversed(blocks):
            if "def " in blk:
                return blk.strip()
        return blocks[-1].strip()
    # fallback
    if "```" in text:
        return text.split("```")[0].strip()
    return text.strip()

def extract_code_from_choices(choices: List[Dict[str, Any]]) -> str:
    if not choices:
        return ""
    turns = choices[0].get("turns", [])
    if isinstance(turns, list):
        text = "\n".join([t if isinstance(t, str) else str(t) for t in turns])
    else:
        text = str(turns)
    return extract_code_from_text(text)

def read_preds_and_map_to_tasks(
    pred_path: str, qid2task: Dict[int, str]
) -> List[Dict[str, str]]:
    """
    Read your predictions JSONL and convert to [{"task_id","completion"}].
    Deduplicate by task_id: keep the FIRST occurrence (change to keep last if needed).
    """
    task_to_completion: Dict[str, str] = {}
    total_lines, mapped = 0, 0

    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            obj = json.loads(line)

            # question_id may be int or numeric string
            qid = obj.get("question_id", obj.get("task_id", None))
            if isinstance(qid, str) and qid.isdigit():
                qid = int(qid)
            if not isinstance(qid, int):
                continue

            task_id = qid2task.get(qid)
            if not task_id:
                continue

            code = extract_code_from_choices(obj.get("choices", []))
            if not code:
                continue

            # 去重策略：保留第一次出现（若想保留最后一次，把条件改成"总是覆盖"）
            if task_id not in task_to_completion:
                task_to_completion[task_id] = code
                mapped += 1
            # else:
            #     task_to_completion[task_id] = code  # 改为保留最后一次

    print(f"[Info] Read {total_lines} lines, mapped {mapped} attempted tasks.")
    samples = [{"task_id": k, "completion": v} for k, v in task_to_completion.items()]
    return samples

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_filtered_problem_file(
    all_problems: Dict[str, Any], attempted_task_ids: List[str], out_path: str
) -> str:
    """
    Write a filtered HumanEval problem file containing ONLY attempted tasks.
    Plain .jsonl is fine (read_problems supports it).
    """
    subset = [all_problems[tid] for tid in attempted_task_ids if tid in all_problems]
    if not subset:
        raise RuntimeError("No attempted tasks found inside the official problem set.")

    write_jsonl(out_path, subset)
    return out_path


def evaluate_single_file(
    pred_path: str,
    save_dir: str,
    all_problems: Dict[str, Any],
    qid2task: Dict[int, str],
    evaluate_func,
    timeout: int = 7,
    n_workers: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate a single prediction file.
    Returns a dict with keys: filename, pass@1, correct, total, error (if any).
    """
    filename = os.path.basename(pred_path)
    result = {"filename": filename, "pass@1": None, "correct": None, "total": None, "error": None}
    
    try:
        # 1) 读取预测，映射到 task_id，并去重
        samples = read_preds_and_map_to_tasks(pred_path, qid2task)
        if not samples:
            result["error"] = "No valid predictions found"
            return result
        attempted_task_ids = [s["task_id"] for s in samples]

        # 2) 写标准 samples.jsonl
        os.makedirs(save_dir, exist_ok=True)
        converted_path = os.path.join(save_dir, "samples.jsonl")
        write_jsonl(converted_path, samples)

        # 3) 写"过滤后的题库"文件（只包含已作答的题）
        filtered_problem_path = os.path.join(save_dir, "filtered_humaneval.jsonl")
        build_filtered_problem_file(all_problems, attempted_task_ids, filtered_problem_path)

        # 4) 评测
        try:
            score = evaluate_func(
                sample_file=converted_path,
                problem_file=filtered_problem_path,
                timeout=timeout,
                n_workers=n_workers,
            )
        except TypeError:
            try:
                score = evaluate_func(
                    sample_file=converted_path,
                    problem_file=filtered_problem_path,
                )
            except TypeError:
                score = evaluate_func(sample_file=converted_path)

        # 5) 解析结果
        total = len(samples)
        acc = None
        correct = None
        if isinstance(score, dict):
            if "pass@1" in score:
                acc = float(score["pass@1"]); correct = int(round(acc * total))
            elif "accuracy" in score:
                acc = float(score["accuracy"]); correct = int(round(acc * total))
            elif "num_correct" in score:
                correct = int(score["num_correct"]); acc = correct / total if total else 0.0
            else:
                for k in ["pass1", "pass_at_1"]:
                    if k in score:
                        acc = float(score[k]); correct = int(round(acc * total)); break

        result["pass@1"] = acc
        result["correct"] = correct
        result["total"] = total

        # 6) 保存单个文件结果
        with open(os.path.join(save_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "pass@1": acc,
                "correct": correct,
                "total": total,
                "raw": score
            }, ensure_ascii=False))

    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing humaneval*.jsonl files (batch mode).")
    parser.add_argument("--pred_path", type=str, default=None,
                        help="Single generated JSONL file path (legacy single-file mode).")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Where to dump converted samples.jsonl and results.")
    parser.add_argument("--project_root", type=str, default="/work/xinyu/MARS/mars/eval",
                        help="Project root so that utils.* can be imported.")
    parser.add_argument("--timeout", type=int, default=7, help="Per test timeout seconds.")
    parser.add_argument("--n-workers", type=int, default=8, help="Parallel workers for evaluation.")
    args = parser.parse_args()

    # Validate arguments
    if not args.input_dir and not args.pred_path:
        parser.error("Either --input_dir or --pred_path is required.")
    if args.input_dir and args.pred_path:
        parser.error("Cannot specify both --input_dir and --pred_path.")

    add_project_root_to_syspath(args.project_root)
    from utils.evaluation import evaluate_functional_correctness

    # 读全量题库 & 构建映射
    all_problems = load_all_problems(PROBLEM_FILE)
    qid2task = build_qid2task_map(all_problems)

    if args.pred_path:
        # Single file mode (legacy)
        result = evaluate_single_file(
            pred_path=args.pred_path,
            save_dir=args.save_dir,
            all_problems=all_problems,
            qid2task=qid2task,
            evaluate_func=evaluate_functional_correctness,
            timeout=args.timeout,
            n_workers=args.n_workers,
        )
        if result["error"]:
            print(f"[Error] {result['error']}")
        elif result["pass@1"] is not None:
            print(f"HumanEval pass@1: {result['pass@1']:.4f}  ({result['correct']}/{result['total']})")
        else:
            print(f"[Info] Converted {result['total']} samples")
    else:
        # Batch mode - process all humaneval*.jsonl files
        pattern = os.path.join(args.input_dir, "humaneval*.jsonl")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"[Error] No files matching 'humaneval*.jsonl' found in {args.input_dir}")
            return
        
        print(f"[Info] Found {len(files)} files to evaluate:")
        for f in files:
            print(f"  - {os.path.basename(f)}")
        print()

        results = []
        for pred_file in files:
            filename = os.path.basename(pred_file)
            # 为每个文件创建独立的输出目录
            file_save_dir = os.path.join(args.save_dir, os.path.splitext(filename)[0])
            
            print(f"[Evaluating] {filename} ...")
            result = evaluate_single_file(
                pred_path=pred_file,
                save_dir=file_save_dir,
                all_problems=all_problems,
                qid2task=qid2task,
                evaluate_func=evaluate_functional_correctness,
                timeout=args.timeout,
                n_workers=args.n_workers,
            )
            results.append(result)
            
            if result["error"]:
                print(f"  [Error] {result['error']}")
            elif result["pass@1"] is not None:
                print(f"  pass@1: {result['pass@1']:.4f}  ({result['correct']}/{result['total']})")
            print()

        # 汇总结果
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Filename':<50} {'pass@1':>10} {'Correct':>10} {'Total':>10}")
        print("-" * 80)
        
        for r in results:
            if r["error"]:
                print(f"{r['filename']:<50} {'ERROR':>10} {'-':>10} {'-':>10}")
            elif r["pass@1"] is not None:
                print(f"{r['filename']:<50} {r['pass@1']:>10.4f} {r['correct']:>10} {r['total']:>10}")
            else:
                print(f"{r['filename']:<50} {'N/A':>10} {'-':>10} {r['total'] or '-':>10}")
        
        print("-" * 80)
        
        # 计算并打印平均值
        valid_results = [r for r in results if r["pass@1"] is not None]
        if valid_results:
            avg_pass1 = sum(r["pass@1"] for r in valid_results) / len(valid_results)
            total_correct = sum(r["correct"] for r in valid_results)
            total_total = sum(r["total"] for r in valid_results)
            print(f"{'AVERAGE':<50} {avg_pass1:>10.4f} {total_correct:>10} {total_total:>10}")
        
        # 保存汇总结果
        summary_path = os.path.join(args.save_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[Info] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

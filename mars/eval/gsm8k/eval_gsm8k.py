#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare predicted answers (#### <final_answer>) with EA reference answers.

Usage (batch mode - recommended):
python /work/xinyu/MARS/mars/eval/gsm8k/eval_gsm8k.py \
  --input_dir /work/xinyu/MARS/fast-llm-inference/speculative_decoding/results/qwen3-0.6b-32b-spd \
  --ref /work/xinyu/MARS/mars/data_2/gsm8k/question.jsonl \
  --save_dir /work/xinyu/MARS/mars/output/gsm8k


Usage (single file mode - legacy):
python eval_gsm8k.py \
  --pred /path/to/gsm8k_xxx.jsonl \
  --ref /path/to/gsm8k/question.jsonl \
  --show-mismatches

Outputs overall accuracy and lists mismatches.
"""

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, Tuple, Optional, Any, List


# —— 数字 token：至少含一位数字 ——
# 1) 金额/整数/带千分位/小数：-12,345.67 或 123 或 3.14
_NUM_TOKEN = r"""
    \$?\s*                                # 可选 $
    (                                     # 统一捕获为 group 1
      (?:-?                               
         (?:
            (?:\d{1,3}(?:,\d{3})+)        # 1,234 或 12,345,678
            | \d+                         # 1234
         )
         (?:\.\d+)?                       # 可选 .56
      ) 
      |
      (?:-?\d+/\d+)                       # 简单分数 -7/8
    )
"""

# final_answer 标签
_TAG = r"<\s*final_answer\s*>"

# 预编译（带注释、忽略大小写、多行不需要）
RE_NUM_TOKEN = re.compile(_NUM_TOKEN, re.VERBOSE)
RE_NUM_TOKEN_I = re.compile(_NUM_TOKEN, re.VERBOSE | re.IGNORECASE)
RE_TAG_I = re.compile(_TAG, re.IGNORECASE)
RE_HASH_LINE = re.compile(r"####\s*([^\n#]+)")

def _clean_num(s: str) -> str:
    return s.replace("$", "").replace(",", "").strip()

def extract_final_answer(text: str) -> Optional[str]:
    """
    提取最终答案字符串（不带单位/货币符号），优先级：
      1) 若出现 <final_answer>：先找其后的第一个"数字 token"；若没有，再找其前最近的"数字 token"。
      2) 若无标签：尝试 '#### N'。
      3) 兜底：全文最后一个"数字 token"。
    """
    if not text:
        return None

    # 1) 先看是否有 <final_answer> 标签（取最后一个最稳）
    tags = list(RE_TAG_I.finditer(text))
    if tags:
        tag = tags[-1]
        s, e = tag.span()

        # 1a) 标签后第一个有效数字
        m_after = RE_NUM_TOKEN_I.search(text, pos=e)
        if m_after:
            return _clean_num(m_after.group(1))

        # 1b) 标签前最近的有效数字（从头到 s 遍历，取最后一个）
        m_before = None
        for m in RE_NUM_TOKEN.finditer(text, 0, s):
            m_before = m
        if m_before:
            return _clean_num(m_before.group(1))
        # 有标签但前后都没有数字 → 继续无标签逻辑

    # 2) 无标签：'#### N'
    m_hash = RE_HASH_LINE.search(text)
    if m_hash:
        return _clean_num(m_hash.group(1))

    # 3) 兜底：全文最后一个有效数字
    nums = list(RE_NUM_TOKEN.finditer(text))
    if nums:
        return _clean_num(nums[-1].group(1))

    return None


def normalize_number(s: str) -> Optional[float]:
    """
    将字符串数值统一转为 float。
    支持:
      - 千分位逗号: 1,234 -> 1234
      - 百分号: 12% -> 0.12
      - 分数: 7/8 -> 0.875
      - 纯数字/小数: 25, 3.14
    仅用于比较数值题；若无法解析返回 None。
    """
    if s is None:
        return None
    st = s.strip()
    if not st:
        return None

    # 百分比
    if st.endswith("%"):
        try:
            return float(st[:-1].replace(",", "").strip()) / 100.0
        except:
            return None

    # 分数 x/y
    if "/" in st and not any(ch.isalpha() for ch in st):
        parts = st.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0].replace(",", "").strip())
                den = float(parts[1].replace(",", "").strip())
                if den == 0:
                    return None
                return num / den
            except:
                pass

    # 普通数字
    try:
        return float(st.replace(",", ""))
    except:
        return None

def load_reference(ref_path: str) -> Dict[int, str]:
    """
    读取 EA 参考文件：每行含 question_id, reference(list)。
    取 reference 最后一行的 '#### <ans>'；如果没 ####，取最后一行。
    返回 {qid: 原始答案字符串}
    """
    ref = {}
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj["question_id"]
            ref_lines = obj.get("reference", [])
            gold_line = ref_lines[-1].strip() if ref_lines else ""
            gold = extract_final_answer(gold_line) or gold_line
            ref[int(qid)] = gold
    return ref

def load_predictions(pred_path: str, choice_index: int = 0) -> Dict[int, str]:
    """
    读取预测文件：
    - 格式A（评测输出）：{"question_id":..., "choices":[{"turns":["..."]}, ...]}
    - 格式B（EA风格）：{"question_id":..., "turns":["..."]}
    返回 {qid: 提取到的最终答案字符串}
    """
    pred = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = int(obj["question_id"])

            # 预测文本
            text = None
            if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                idx = min(max(choice_index, 0), len(obj["choices"]) - 1)
                turns = obj["choices"][idx].get("turns", [])
                if isinstance(turns, list) and turns:
                    text = turns[-1]
            elif "turns" in obj and isinstance(obj["turns"], list) and obj["turns"]:
                text = obj["turns"][-1]

            ans = extract_final_answer(text or "")
            pred[qid] = ans
    return pred

def compare_answers(gold: str, pred: str, atol: float = 1e-6) -> Tuple[bool, str]:
    """
    比较两个答案。优先数值比较；数值解析失败则做严格字符串比较（去除首尾空格）。
    返回 (是否正确, 说明)
    """
    g_num = normalize_number(gold)
    p_num = normalize_number(pred)
    if g_num is not None and p_num is not None:
        ok = math.isclose(g_num, p_num, rel_tol=0.0, abs_tol=atol)
        reason = f"num gold={g_num} pred={p_num}"
        return ok, reason
    # 退化为字符串比较（常见于单位或符号答案）
    ok = (str(gold).strip() == str(pred).strip())
    reason = "str match"
    return ok, reason


def evaluate_single_file(
    pred_path: str,
    ref: Dict[int, str],
    choice_index: int = 0,
    show_mismatches: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single prediction file against reference.
    Returns a dict with keys: filename, accuracy, correct, total, error (if any).
    """
    filename = os.path.basename(pred_path)
    result = {"filename": filename, "accuracy": None, "correct": None, "total": None, "error": None}
    
    try:
        pred = load_predictions(pred_path, choice_index=choice_index)
        
        total = 0
        correct = 0
        mismatches = []
        
        for qid, gold in ref.items():
            total += 1
            p = pred.get(qid, None)
            if p is None:
                mismatches.append((qid, gold, None, "missing prediction"))
                continue
            ok, why = compare_answers(gold, p)
            if ok:
                correct += 1
            else:
                mismatches.append((qid, gold, p, why))
        
        acc = correct / total if total else 0.0
        result["accuracy"] = acc
        result["correct"] = correct
        result["total"] = total
        result["mismatches"] = mismatches
        
        if show_mismatches and mismatches:
            print(f"\n  Mismatches for {filename}:")
            for qid, gold, p, why in mismatches[:20]:  # 限制输出
                print(f"    [qid={qid}] gold='{gold}'  pred='{p}'  ({why})")
            if len(mismatches) > 20:
                print(f"    ... and {len(mismatches) - 20} more")
                
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default=None,
                    help="Directory containing gsm8k*.jsonl files (batch mode).")
    ap.add_argument("--pred", type=str, default=None,
                    help="Single predictions jsonl path (legacy single-file mode).")
    ap.add_argument("--ref", default="/work/xinyu/MARS/mars/data_1/gsm8k/question.jsonl",
                    help="Reference EA jsonl path")
    ap.add_argument("--save_dir", type=str, default=None,
                    help="Where to save summary results (optional).")
    ap.add_argument("--choice-index", type=int, default=0,
                    help="Which choice to grade (default=0)")
    ap.add_argument("--show-mismatches", action="store_true",
                    help="Print mismatched cases")
    args = ap.parse_args()

    # Validate arguments
    if not args.input_dir and not args.pred:
        ap.error("Either --input_dir or --pred is required.")
    if args.input_dir and args.pred:
        ap.error("Cannot specify both --input_dir and --pred.")

    # Load reference once
    ref = load_reference(args.ref)

    if args.pred:
        # Single file mode (legacy)
        result = evaluate_single_file(
            pred_path=args.pred,
            ref=ref,
            choice_index=args.choice_index,
            show_mismatches=args.show_mismatches,
        )
        if result["error"]:
            print(f"[Error] {result['error']}")
        else:
            print(f"Total: {result['total']}")
            print(f"Correct: {result['correct']}")
            print(f"Accuracy: {result['accuracy']:.4f}")
    else:
        # Batch mode - process all gsm8k*.jsonl files
        pattern = os.path.join(args.input_dir, "gsm8k*.jsonl")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"[Error] No files matching 'gsm8k*.jsonl' found in {args.input_dir}")
            return
        
        print(f"[Info] Found {len(files)} files to evaluate:")
        for f in files:
            print(f"  - {os.path.basename(f)}")
        print()

        results = []
        for pred_file in files:
            filename = os.path.basename(pred_file)
            
            print(f"[Evaluating] {filename} ...")
            result = evaluate_single_file(
                pred_path=pred_file,
                ref=ref,
                choice_index=args.choice_index,
                show_mismatches=args.show_mismatches,
            )
            results.append(result)
            
            if result["error"]:
                print(f"  [Error] {result['error']}")
            elif result["accuracy"] is not None:
                print(f"  Accuracy: {result['accuracy']:.4f}  ({result['correct']}/{result['total']})")
            print()

        # 汇总结果
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Filename':<50} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
        print("-" * 80)
        
        for r in results:
            if r["error"]:
                print(f"{r['filename']:<50} {'ERROR':>10} {'-':>10} {'-':>10}")
            elif r["accuracy"] is not None:
                print(f"{r['filename']:<50} {r['accuracy']:>10.4f} {r['correct']:>10} {r['total']:>10}")
            else:
                print(f"{r['filename']:<50} {'N/A':>10} {'-':>10} {r['total'] or '-':>10}")
        
        print("-" * 80)
        
        # 计算并打印平均值
        valid_results = [r for r in results if r["accuracy"] is not None]
        if valid_results:
            avg_acc = sum(r["accuracy"] for r in valid_results) / len(valid_results)
            total_correct = sum(r["correct"] for r in valid_results)
            total_total = sum(r["total"] for r in valid_results)
            print(f"{'AVERAGE':<50} {avg_acc:>10.4f} {total_correct:>10} {total_total:>10}")
        
        # 保存汇总结果
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            summary_path = os.path.join(args.save_dir, "gsm8k_summary.json")
            # 移除 mismatches 以减小文件大小
            summary_results = []
            for r in results:
                r_copy = {k: v for k, v in r.items() if k != "mismatches"}
                summary_results.append(r_copy)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_results, f, ensure_ascii=False, indent=2)
            print(f"\n[Info] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

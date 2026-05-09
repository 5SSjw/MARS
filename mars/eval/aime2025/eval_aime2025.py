#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare predicted answers (#### <final_answer> or \\boxed{...}) with EA reference answers.

Usage:
python3 /workspace/tmp1/MARS/mars/eval/aime2025/eval_aime2025.py \
  --pred /workspace/tmp1/MARS/mars/outputs/test_1207/aime2025_qwen3-8b_ea_non-thinking_t1.0_b0.8_r0.9.jsonl \
  --ref  /workspace/tmp1/MARS/mars/data/aime2025/question.jsonl
  --show-mismatches

Outputs overall accuracy and lists mismatches.
"""

import argparse
import json
import math
import re
from typing import Dict, Tuple, Optional


# ---- 抽取 \boxed{...} -------------------------------------------------------

def extract_boxed_answer(solution: str) -> Optional[str]:
    """
    Extract the answer from the last \\boxed{} in a solution string.
    Handles various edge cases and formats.
    
    Args:
        solution (str): The solution string containing \\boxed{} expressions
        
    Returns:
        str: The extracted answer (inside \\boxed{}), or None if no valid \\boxed found
    """
    if not solution:
        return None
    
    # 保留你原来对 "ANSWER:" 的兼容逻辑（如果不用可以删掉这一段）
    if "ANSWER:" in solution:
        answer = solution.split("ANSWER:")[-1].strip()
        if len(answer) > 1:
            answer = answer[0]
        return answer

    # 找最后一个 '\boxed'
    idx = solution.rfind(r'\boxed')
    if idx == -1:
        return None

    # 找紧跟其后的第一个 '{'
    start_brace = solution.find('{', idx)
    if start_brace == -1:
        return None

    # 用括号计数法找到与之匹配的 '}'
    depth = 0
    end_brace = None
    for i, ch in enumerate(solution[start_brace:], start=start_brace):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_brace = i
                break

    if end_brace is None:
        return None

    content = solution[start_brace + 1 : end_brace].strip()
    if not content:
        return None

    # 去掉包在 $...$ 里的情况：\boxed{$123$}
    if content.startswith('$') and content.endswith('$') and len(content) > 2:
        content = content[1:-1].strip()

    return content


# ---- 数值归一化 & 比较 -------------------------------------------------------

# 简单处理 \frac{a}{b}
FRAC_PATTERN = re.compile(r'\\frac\{([^{}]+)\}\{([^{}]+)\}')


def normalize_number(s: str) -> Optional[float]:
    """
    将字符串数值统一转为 float，用于数值比较。
    支持:
      - 纯数字/小数: 25, 3.14
      - 千分位逗号: 1,234 -> 1234
      - 百分号: 12% -> 0.12
      - 分数: 7/8 -> 0.875
      - LaTeX: \\frac{7}{8} -> 0.875
    无法解析则返回 None。
    """
    if s is None:
        return None
    st = s.strip()
    if not st:
        return None

    # LaTeX 分数 \frac{a}{b}
    m = FRAC_PATTERN.fullmatch(st)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        try:
            num = float(a.replace(",", ""))
            den = float(b.replace(",", ""))
            if den == 0:
                return None
            return num / den
        except:
            return None

    # 百分号
    if st.endswith("%"):
        try:
            return float(st[:-1].replace(",", "").strip()) / 100.0
        except:
            return None

    # 文本分数 a/b（不含字母）
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

    # 退化为字符串比较（适用于非纯数字答案）
    ok = (str(gold).strip() == str(pred).strip())
    reason = "str match"
    return ok, reason


# ---- 读取 ref 和 pred -------------------------------------------------------

def load_reference(ref_path: str) -> Dict[int, str]:
    """
    读取参考文件（EA 转换后的新格式）：
      每行 JSON 形如：
        {
          "question_id": 0,
          "category": "math",
          "turns": [...],
          "reference": ["70"]
        }
    这里直接取 reference 的最后一项作为 gold（不再解析 #### 或 boxed）。
    返回: {qid: gold_string}
    """
    ref = {}
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = int(obj["question_id"])
            ref_lines = obj.get("reference", [])
            gold = str(ref_lines[-1]).strip() if ref_lines else ""
            ref[qid] = gold
    return ref


def load_predictions(pred_path: str, choice_index: int = 0) -> Dict[int, str]:
    """
    读取预测文件：
    - 格式A：{"question_id":..., "choices":[{"turns":["..."]}, ...]}
    - 格式B：{"question_id":..., "turns":["..."]}
    从最后一条回复中抽取 \\boxed{...} 的内容。
    返回 {qid: boxed_inner_string}
    """
    pred = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = int(obj["question_id"])

            text = None
            if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                idx = min(max(choice_index, 0), len(obj["choices"]) - 1)
                turns = obj["choices"][idx].get("turns", [])
                if isinstance(turns, list) and turns:
                    text = turns[-1]
            elif "turns" in obj and isinstance(obj["turns"], list) and obj["turns"]:
                text = obj["turns"][-1]

            ans = extract_boxed_answer(text or "")
            pred[qid] = ans
    return pred


# ---- 主程序 -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Predictions jsonl path")
    ap.add_argument("--ref", required=True, help="Reference jsonl path")
    ap.add_argument("--choice-index", type=int, default=0, help="Which choice to grade (default=0)")
    ap.add_argument("--show-mismatches", action="store_true", help="Print mismatched cases")
    args = ap.parse_args()

    ref = load_reference(args.ref)
    pred = load_predictions(args.pred, choice_index=args.choice_index)

    total = 0
    correct = 0
    mismatches = []

    for qid, gold in ref.items():
        total += 1
        p = pred.get(qid, None)
        if p is None:
            mismatches.append((qid, gold, None, "missing prediction (no \\boxed found)"))
            continue
        ok, why = compare_answers(gold, p)
        if ok:
            correct += 1
        else:
            mismatches.append((qid, gold, p, why))

    acc = correct / total if total else 0.0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")

    if args.show_mismatches and mismatches:
        print("\nMismatches:")
        for qid, gold, p, why in mismatches[:200]:  # 防爆屏
            print(f"[qid={qid}] gold='{gold}'  pred='{p}'  ({why})")


if __name__ == "__main__":
    main()
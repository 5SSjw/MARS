#!/bin/bash
set -e

# ====== Paths & Environment ======
REPO_ROOT="/work/xinyu/MARS"
cd "$REPO_ROOT"

# 清理 Ray 状态
ray stop --force >/dev/null 2>&1 || true
unset RAY_HEAD_IP RAY_CLUSTER
export RAY_ADDRESS=local

# GPU 与 Python 路径
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# ====== Model & Params ======
# BASE_MODEL_PATH="/work/xinyu/models/Qwen3-8B"
# EA_MODEL_PATH="/work/xinyu/models/qwen3_8b_mars"

# MODEL_ID="qwen3-8b"
BASE_MODEL_PATH="/work/xinyu/models/Qwen3-32B"
EA_MODEL_PATH="/work/xinyu/models/Qwen3-32B_mars"

MODEL_ID="qwen3-32b"

# 要遍历的参数数组
TEMPS=(1.0)
DEPTHS=(6)
BENCH_NAMES=("wmt19")
# "humaneval" "alpaca" "gsm8k" "mt_bench" "sum"

# Relaxation thresholds for evaluate_posterior (可遍历多个阈值)
RELAXATION_THRESHOLDS=(0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98) 

# 计算总任务数
TOTAL=$((${#TEMPS[@]} * ${#DEPTHS[@]} * ${#BENCH_NAMES[@]} * ${#RELAXATION_THRESHOLDS[@]}))
CURRENT=0

# ====== Run EA Evaluation for all combinations ======
for TEMP in "${TEMPS[@]}"; do
  for DEPTH in "${DEPTHS[@]}"; do
    for BENCH_NAME in "${BENCH_NAMES[@]}"; do
      for RELAX_TH in "${RELAXATION_THRESHOLDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo "======================================"
        echo "[${CURRENT}/${TOTAL}] Running: ${BENCH_NAME} | depth=${DEPTH} | temp=${TEMP} | relax_th=${RELAX_TH}"
        echo "======================================"
        
        # 创建输出目录
        OUTPUT_DIR="$REPO_ROOT/mars/rebuttal/${MODEL_ID}"
        mkdir -p "${OUTPUT_DIR}"
        
        OUTPUT_PATH="${OUTPUT_DIR}/${BENCH_NAME}_d${DEPTH}_t${TEMP}_ratio${RELAX_TH}.jsonl"
        
        python -m mars.evaluation.gen_ea_answer_qwen3 \
          --base-model-path "${BASE_MODEL_PATH}" \
          --ea-model-path "${EA_MODEL_PATH}" \
          --model-id "${MODEL_ID}" \
          --bench-name "${BENCH_NAME}" \
          --answer-file "${OUTPUT_PATH}" \
          --num-gpus-total 1 \
          --num-gpus-per-model 1 \
          --temperature "${TEMP}" \
          --num-choices 1 \
          --total-token 50 \
          --depth "${DEPTH}" \
          --top-k 10 \
          --use-mars \
          --disable-thinking \
          --relaxation-threshold "${RELAX_TH}"
        
        echo "Finished: ${BENCH_NAME} | depth=${DEPTH} | temp=${TEMP} | relax_th=${RELAX_TH}"
        echo ""
      done
    done
  done
done

echo "All ${TOTAL} experiments completed!"
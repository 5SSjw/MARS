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
BASE_MODEL_PATH="/work/xinyu/models/vicuna-13b-v1.3"
EA_MODEL_PATH="/work/xinyu/models/MARS-Vicuna1.3-13B"

MODEL_ID="vicuna13b"
TEMP=1.0
DEPTH=6

# 要运行的多个数据集
BENCH_NAMES=("wmt19")

# Relaxation thresholds for evaluate_posterior (可遍历多个阈值)
RELAXATION_THRESHOLDS=(0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98) 

# 计算总任务数
TOTAL=$((${#BENCH_NAMES[@]} * ${#RELAXATION_THRESHOLDS[@]}))
CURRENT=0

# ====== Run EA Evaluation for each benchmark ======
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
    
    python -m mars.evaluation.gen_ea_answer_vicuna \
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
      --relaxation-threshold "${RELAX_TH}"
    
    echo "Finished: ${BENCH_NAME} | relax_th=${RELAX_TH}"
    echo ""
  done
done

echo "All ${TOTAL} experiments completed!"
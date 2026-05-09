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
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# ====== Model & Params ======
BASE_MODEL_PATH="/work/xinyu/models/DeepSeek-R1-Distill-Llama-8B"
EA_MODEL_PATH="/work/xinyu/models/MARS-DeepSeek-R1-Distill-LLaMA-8B"

MODEL_ID="dpsk-llama-8b"
TEMP=1.0
DEPTH=6

# 要运行的多个数据集
BENCH_NAMES=("humaneval" "alpaca" "gsm8k" "mt_bench" "sum")

# ====== Run EA Evaluation for each benchmark ======
for BENCH_NAME in "${BENCH_NAMES[@]}"; do
  echo "======================================"
  echo "Running benchmark: ${BENCH_NAME}"
  echo "======================================"
  
  # 创建输出目录
  OUTPUT_DIR="$REPO_ROOT/mars/final/${MODEL_ID}"
  mkdir -p "${OUTPUT_DIR}"
  
  OUTPUT_PATH="${OUTPUT_DIR}/${BENCH_NAME}_t${DEPTH}_t${TEMP}_ratio0.9.jsonl"
  
  python -m mars.evaluation.gen_ea_answer_ds \
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
    --use-mars
  
  echo "Finished benchmark: ${BENCH_NAME}"
  echo ""
done

echo "All benchmarks completed!"
#!/bin/bash
set -e

# ====== Paths & Environment ======
REPO_ROOT="/workspace/tmp1/MARS"
cd "$REPO_ROOT"

# 清理 Ray 状态
ray stop --force >/dev/null 2>&1 || true
unset RAY_HEAD_IP RAY_CLUSTER
export RAY_ADDRESS=local

# GPU 与 Python 路径
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# ====== Model & Params ======
BASE_MODEL_PATH="/workspace/models/Llama-3.1-8B-Instruct"
EA_MODEL_PATH="/workspace/models/MARS-LLaMA3.1-Instruct-8B"

MODEL_ID="llama3-8b"
BENCH_NAME="humaneval"
TEMP=1.0

OUTPUT_PATH="$REPO_ROOT/mars/outputs/test_1124/${BENCH_NAME}_${MODEL_ID}_baseline_t${TEMP}.jsonl"

# ====== Run EA Evaluation ======
python -m mars.evaluation.gen_baseline_answer_llama3chat \
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
  --depth 6 \
  --top-k 10

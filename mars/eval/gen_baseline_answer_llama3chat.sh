#!/bin/bash
set -e

REPO_ROOT="YOUR_REPO_ROOT"
cd "$REPO_ROOT"

ray stop --force >/dev/null 2>&1 || true
unset RAY_HEAD_IP RAY_CLUSTER
export RAY_ADDRESS=local    

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

BASE_MODEL_PATH="YOUR_MODEL_PATH/Meta-Llama-3.1-8B-Instruct"
EA_MODEL_PATH="YOUR_MODEL_PATH/MARS-LLaMA3.1-Instruct-8B"
MODEL_ID="llama38b2_40"
BENCH_NAME="humaneval"
OUTPUT_PATH="$REPO_ROOT/mars/outputs/${BENCH_NAME}_${MODEL_ID}_baseline_dp4_gnum8_test.jsonl"

python -m mars.evaluation.gen_baseline_answer_llama3chat \
  --base-model-path "${BASE_MODEL_PATH}" \
  --ea-model-path "${EA_MODEL_PATH}" \
  --model-id "${MODEL_ID}" \
  --bench-name "${BENCH_NAME}" \
  --answer-file "${OUTPUT_PATH}" \
  --num-gpus-total 8 \
  --num-gpus-per-model 1 \
  --temperature 0.0 \
  --num-choices 1 \
  --total-token 60 \
  --depth 5 \
  --top-k 10

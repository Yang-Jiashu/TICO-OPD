#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen3-4B}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000/v1}
N_SAMPLES=${N_SAMPLES:-1}
MAX_TOKENS=${MAX_TOKENS:-16384}
OUT_DIR=${OUT_DIR:-eval_results/qwen3_math}

mkdir -p "${OUT_DIR}"

python3 eval/run_math_eval.py \
  --dataset-name OpenRLHF/aime-2024 \
  --split train \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --n-samples "${N_SAMPLES}" \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --max-tokens "${MAX_TOKENS}" \
  --out "${OUT_DIR}/aime24.jsonl"

python3 eval/run_math_eval.py \
  --dataset-name MathArena/aime_2025 \
  --split train \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --n-samples "${N_SAMPLES}" \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --max-tokens "${MAX_TOKENS}" \
  --out "${OUT_DIR}/aime25.jsonl"

python3 eval/run_math_eval.py \
  --dataset-name HuggingFaceH4/MATH-500 \
  --split test \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --n-samples "${N_SAMPLES}" \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --max-tokens "${MAX_TOKENS}" \
  --out "${OUT_DIR}/math500.jsonl"

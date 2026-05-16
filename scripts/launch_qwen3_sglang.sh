#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-32768}

export CUDA_VISIBLE_DEVICES

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --context-length "${MAX_TOTAL_TOKENS}" \
  --chunked-prefill-size 8192

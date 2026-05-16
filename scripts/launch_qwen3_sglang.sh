#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../configs/qwen3/qwen3_model_matrix.sh"

MODEL_SIZE=${MODEL_SIZE:-4B}
MODEL_PATH=${MODEL_PATH:-$(qwen3_model_id "${MODEL_SIZE}")}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP=${TP:-$(qwen3_default_tp "${MODEL_SIZE}")}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-32768}

export CUDA_VISIBLE_DEVICES

echo "Launching Qwen3 endpoint"
echo "  MODEL_SIZE=${MODEL_SIZE}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  TP=${TP}"
echo "  PORT=${PORT}"

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --context-length "${MAX_TOTAL_TOKENS}" \
  --chunked-prefill-size 8192

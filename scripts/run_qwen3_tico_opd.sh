#!/usr/bin/env bash
set -euo pipefail

# Run from the slime repo root after applying the TICO-OPD patch.
# Teacher and student Qwen3 sizes are independently configurable.
#
# Example:
#   STUDENT_SIZE=4B TEACHER_SIZE=32B bash scripts/run_qwen3_tico_opd.sh
#   STUDENT_SIZE=8B TEACHER_SIZE=235B-A22B TEACHER_TP=8 bash scripts/run_qwen3_tico_opd.sh
#   USE_EXTERNAL_TEACHER=true TEACHER_URL=http://teacher-host:13141/generate bash scripts/run_qwen3_tico_opd.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../configs/qwen3/qwen3_model_matrix.sh"

SLIME_DIR=${SLIME_DIR:-/root/slime}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}

STUDENT_SIZE=${STUDENT_SIZE:-4B}
TEACHER_SIZE=${TEACHER_SIZE:-32B}

STUDENT_MODEL_ID=${STUDENT_MODEL_ID:-$(qwen3_model_id "${STUDENT_SIZE}")}
TEACHER_MODEL_ID=${TEACHER_MODEL_ID:-$(qwen3_model_id "${TEACHER_SIZE}")}

BASE_MODEL=${BASE_MODEL:-/root/Qwen3-${STUDENT_SIZE}}
TEACHER_MODEL=${TEACHER_MODEL:-/root/Qwen3-${TEACHER_SIZE}}
REF_LOAD=${REF_LOAD:-/root/Qwen3-${STUDENT_SIZE}_torch_dist}
SAVE_DIR=${SAVE_DIR:-/root/checkpoints/qwen3-${STUDENT_SIZE}-tico-opd}

TRAIN_DATA=${TRAIN_DATA:-/root/data/math/dapo17k.jsonl}
AIME24=${AIME24:-/root/data/math/aime24.jsonl}
AIME25=${AIME25:-/root/data/math/aime25.jsonl}
MATH500=${MATH500:-/root/data/math/math500.jsonl}

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-${NUM_GPUS}}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-${NUM_GPUS}}
STUDENT_TP=${STUDENT_TP:-$(qwen3_default_tp "${STUDENT_SIZE}")}
TEACHER_TP=${TEACHER_TP:-$(qwen3_default_tp "${TEACHER_SIZE}")}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-$(qwen3_default_max_tokens_per_gpu "${STUDENT_SIZE}")}
ROLLOUT_GPUS_PER_ENGINE=${ROLLOUT_GPUS_PER_ENGINE:-${STUDENT_TP}}

TEACHER_IP=${TEACHER_IP:-127.0.0.1}
TEACHER_PORT=${TEACHER_PORT:-13141}
USE_EXTERNAL_TEACHER=${USE_EXTERNAL_TEACHER:-false}
TEACHER_URL=${TEACHER_URL:-http://${TEACHER_IP}:${TEACHER_PORT}/generate}
TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES:-}
TEACHER_MEM_FRACTION_STATIC=${TEACHER_MEM_FRACTION_STATIC:-0.6}

MODEL_ARGS_FILE="$(qwen3_slime_model_args_file "${STUDENT_SIZE}")"
source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}"

echo "TICO-OPD Qwen3 training"
echo "  student: ${STUDENT_SIZE} (${BASE_MODEL}; hub=${STUDENT_MODEL_ID})"
echo "  teacher: ${TEACHER_SIZE} (${TEACHER_MODEL}; hub=${TEACHER_MODEL_ID})"
echo "  student TP: ${STUDENT_TP}"
echo "  teacher TP: ${TEACHER_TP}"
echo "  teacher URL: ${TEACHER_URL}"
echo "  train data: ${TRAIN_DATA}"

if [[ "${USE_EXTERNAL_TEACHER}" == "true" ]]; then
  echo "Using external teacher server: ${TEACHER_URL}"
else
  TEACHER_ENV=()
  if [[ -n "${TEACHER_CUDA_VISIBLE_DEVICES}" ]]; then
    TEACHER_ENV=(env CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}")
    echo "Starting local teacher with CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES}"
  else
    echo "Starting local teacher without overriding CUDA_VISIBLE_DEVICES."
    echo "Set TEACHER_CUDA_VISIBLE_DEVICES to avoid teacher/student GPU overlap on a shared node."
  fi

  "${TEACHER_ENV[@]}" python3 -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" \
    --host 0.0.0.0 \
    --port "${TEACHER_PORT}" \
    --tp "${TEACHER_TP}" \
    --chunked-prefill-size 8192 \
    --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" &

  until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; do
    echo "waiting for teacher at ${TEACHER_URL}"
    sleep 5
  done
fi

ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${BASE_MODEL}" \
  --ref-load "${REF_LOAD}" \
  --load "${SAVE_DIR}" \
  --save "${SAVE_DIR}" \
  --save-interval 20 \
  --prompt-data "${TRAIN_DATA}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --num-rollout 3000 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 0.6 \
  --global-batch-size 256 \
  --balance-data \
  --eval-interval 20 \
  --eval-prompt-data aime24 "${AIME24}" aime25 "${AIME25}" math500 "${MATH500}" \
  --n-samples-per-eval-prompt 16 \
  --eval-max-response-len 16384 \
  --eval-top-p 0.95 \
  --tensor-model-parallel-size "${STUDENT_TP}" \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --advantage-estimator grpo \
  --use-opd \
  --opd-type sglang \
  --opd-kl-coef 1.0 \
  --policy-loss-type future_kl \
  --future-kl-decay-rate 32 \
  --future-kl-start include_current \
  --future-kl-window -1 \
  --future-kl-clip-ratio 0.2 \
  --future-kl-clip-high-only \
  --future-kl-safety-threshold 10.0 \
  --eps-clip-c 3.0 \
  --use-compression-opd \
  --compression-length-budget-ratio 0.75 \
  --compression-advantage-coef 0.02 \
  --compression-eos-coef 0.01 \
  --compression-coverage-threshold 0.90 \
  --compression-importance-decay-rate 32 \
  --compression-reward-coef 0.0 \
  --use-kl-loss \
  --kl-loss-coef 0.00 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.00 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}" \
  --sglang-mem-fraction-static 0.7 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func \
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
  --rm-url "${TEACHER_URL}"

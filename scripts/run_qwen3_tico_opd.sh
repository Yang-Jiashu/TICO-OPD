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

SLIME_DIR=${SLIME_DIR:-$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}

STUDENT_SIZE=${STUDENT_SIZE:-4B}
TEACHER_SIZE=${TEACHER_SIZE:-32B}

STUDENT_MODEL_ID=${STUDENT_MODEL_ID:-$(qwen3_model_id "${STUDENT_SIZE}")}
TEACHER_MODEL_ID=${TEACHER_MODEL_ID:-$(qwen3_model_id "${TEACHER_SIZE}")}

BASE_MODEL=${BASE_MODEL:-/root/Qwen3-${STUDENT_SIZE}}
TEACHER_MODEL=${TEACHER_MODEL:-/root/Qwen3-${TEACHER_SIZE}}
REF_LOAD=${REF_LOAD:-/root/Qwen3-${STUDENT_SIZE}_torch_dist}
SAVE_DIR=${SAVE_DIR:-/root/checkpoints/qwen3-${STUDENT_SIZE}-tico-opd}

TRAIN_DATA=${TRAIN_DATA:-${SLIME_DIR}/data/math/dapo17k.jsonl}
AIME24=${AIME24:-${SLIME_DIR}/data/math/aime24.jsonl}
AIME25=${AIME25:-${SLIME_DIR}/data/math/aime25.jsonl}
MATH500=${MATH500:-${SLIME_DIR}/data/math/math500.jsonl}

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

OPD_KL_COEF=${OPD_KL_COEF:-1.0}
POLICY_LOSS_TYPE=${POLICY_LOSS_TYPE:-future_kl}
FUTURE_KL_DECAY_RATE=${FUTURE_KL_DECAY_RATE:-32}
FUTURE_KL_START=${FUTURE_KL_START:-include_current}
FUTURE_KL_WINDOW=${FUTURE_KL_WINDOW:--1}
FUTURE_KL_CLIP_RATIO=${FUTURE_KL_CLIP_RATIO:-0.2}
FUTURE_KL_SAFETY_THRESHOLD=${FUTURE_KL_SAFETY_THRESHOLD:-10.0}
FUTURE_KL_AVERAGE=${FUTURE_KL_AVERAGE:-false}
FUTURE_KL_CLIP_HIGH_ONLY=${FUTURE_KL_CLIP_HIGH_ONLY:-true}

USE_COMPRESSION_OPD=${USE_COMPRESSION_OPD:-true}
COMPRESSION_LENGTH_BUDGET=${COMPRESSION_LENGTH_BUDGET:--1}
COMPRESSION_LENGTH_BUDGET_RATIO=${COMPRESSION_LENGTH_BUDGET_RATIO:-0.75}
COMPRESSION_ADVANTAGE_COEF=${COMPRESSION_ADVANTAGE_COEF:-0.02}
COMPRESSION_EOS_COEF=${COMPRESSION_EOS_COEF:-0.01}
COMPRESSION_COVERAGE_THRESHOLD=${COMPRESSION_COVERAGE_THRESHOLD:-0.90}
COMPRESSION_MIN_RESPONSE_LEN=${COMPRESSION_MIN_RESPONSE_LEN:-0}
COMPRESSION_IMPORTANCE_DECAY_RATE=${COMPRESSION_IMPORTANCE_DECAY_RATE:-32}
COMPRESSION_IMPORTANCE_START=${COMPRESSION_IMPORTANCE_START:-include_current}
COMPRESSION_IMPORTANCE_WINDOW=${COMPRESSION_IMPORTANCE_WINDOW:--1}
COMPRESSION_IMPORTANCE_AVERAGE=${COMPRESSION_IMPORTANCE_AVERAGE:-false}
COMPRESSION_IMPORTANCE_TEMPERATURE=${COMPRESSION_IMPORTANCE_TEMPERATURE:-1.0}
COMPRESSION_REWARD_COEF=${COMPRESSION_REWARD_COEF:-0.0}
COMPRESSION_MASK_LOW_IMPORTANCE_TOKENS=${COMPRESSION_MASK_LOW_IMPORTANCE_TOKENS:-false}
COMPRESSION_LOW_IMPORTANCE_THRESHOLD=${COMPRESSION_LOW_IMPORTANCE_THRESHOLD:-0.2}

MODEL_ARGS_FILE="$(qwen3_slime_model_args_file "${STUDENT_SIZE}")"
source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}"

echo "TICO-OPD Qwen3 training"
echo "  student: ${STUDENT_SIZE} (${BASE_MODEL}; hub=${STUDENT_MODEL_ID})"
echo "  teacher: ${TEACHER_SIZE} (${TEACHER_MODEL}; hub=${TEACHER_MODEL_ID})"
echo "  student TP: ${STUDENT_TP}"
echo "  teacher TP: ${TEACHER_TP}"
echo "  teacher URL: ${TEACHER_URL}"
echo "  train data: ${TRAIN_DATA}"
echo "  policy loss: ${POLICY_LOSS_TYPE}"
echo "  compression OPD: ${USE_COMPRESSION_OPD}"

EXTRA_TICO_ARGS=()
if [[ "${FUTURE_KL_AVERAGE}" == "true" ]]; then
  EXTRA_TICO_ARGS+=(--future-kl-average)
fi
if [[ "${FUTURE_KL_CLIP_HIGH_ONLY}" == "true" ]]; then
  EXTRA_TICO_ARGS+=(--future-kl-clip-high-only)
else
  EXTRA_TICO_ARGS+=(--no-future-kl-clip-high-only)
fi
if [[ "${USE_COMPRESSION_OPD}" == "true" ]]; then
  EXTRA_TICO_ARGS+=(--use-compression-opd)
fi
if [[ "${COMPRESSION_IMPORTANCE_AVERAGE}" == "true" ]]; then
  EXTRA_TICO_ARGS+=(--compression-importance-average)
fi
if [[ "${COMPRESSION_MASK_LOW_IMPORTANCE_TOKENS}" == "true" ]]; then
  EXTRA_TICO_ARGS+=(--compression-mask-low-importance-tokens)
fi

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
  --opd-kl-coef "${OPD_KL_COEF}" \
  --policy-loss-type "${POLICY_LOSS_TYPE}" \
  --future-kl-decay-rate "${FUTURE_KL_DECAY_RATE}" \
  --future-kl-start "${FUTURE_KL_START}" \
  --future-kl-window "${FUTURE_KL_WINDOW}" \
  --future-kl-clip-ratio "${FUTURE_KL_CLIP_RATIO}" \
  --future-kl-safety-threshold "${FUTURE_KL_SAFETY_THRESHOLD}" \
  --eps-clip-c 3.0 \
  --compression-length-budget "${COMPRESSION_LENGTH_BUDGET}" \
  --compression-length-budget-ratio "${COMPRESSION_LENGTH_BUDGET_RATIO}" \
  --compression-advantage-coef "${COMPRESSION_ADVANTAGE_COEF}" \
  --compression-eos-coef "${COMPRESSION_EOS_COEF}" \
  --compression-coverage-threshold "${COMPRESSION_COVERAGE_THRESHOLD}" \
  --compression-min-response-len "${COMPRESSION_MIN_RESPONSE_LEN}" \
  --compression-importance-decay-rate "${COMPRESSION_IMPORTANCE_DECAY_RATE}" \
  --compression-importance-start "${COMPRESSION_IMPORTANCE_START}" \
  --compression-importance-window "${COMPRESSION_IMPORTANCE_WINDOW}" \
  --compression-importance-temperature "${COMPRESSION_IMPORTANCE_TEMPERATURE}" \
  --compression-reward-coef "${COMPRESSION_REWARD_COEF}" \
  --compression-low-importance-threshold "${COMPRESSION_LOW_IMPORTANCE_THRESHOLD}" \
  "${EXTRA_TICO_ARGS[@]}" \
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

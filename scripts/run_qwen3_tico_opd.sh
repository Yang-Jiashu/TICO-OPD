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
SAVE_INTERVAL=${SAVE_INTERVAL:-20}

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
TEACHER_CHUNKED_PREFILL_SIZE=${TEACHER_CHUNKED_PREFILL_SIZE:-8192}

NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.6}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-grpo}
EPS_CLIP_C=${EPS_CLIP_C:-3.0}
OPTIMIZER=${OPTIMIZER:-adam}
LR=${LR:-1e-6}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-constant}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.98}

EVAL_INTERVAL=${EVAL_INTERVAL:-20}
N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-16}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-16384}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.6}
EVAL_TOP_P=${EVAL_TOP_P:-0.95}
EVAL_TOP_K=${EVAL_TOP_K:-20}

SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.7}

USE_SWANLAB=${USE_SWANLAB:-false}
SWANLAB_MODE=${SWANLAB_MODE:-cloud}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-TICO-OPD}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-qwen3-${STUDENT_SIZE}-teacher-${TEACHER_SIZE}}
SWANLAB_GROUP=${SWANLAB_GROUP:-qwen3-tico-opd}
SWANLAB_TAGS=${SWANLAB_TAGS:-tico-opd,qwen3}
SWANLAB_LOGDIR=${SWANLAB_LOGDIR:-}
SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
SWANLAB_HOST=${SWANLAB_HOST:-}
SWANLAB_RUN_ID=${SWANLAB_RUN_ID:-}
SWANLAB_RESUME=${SWANLAB_RESUME:-}
SWANLAB_PUBLIC=${SWANLAB_PUBLIC:-}

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
echo "  rollout: num=${NUM_ROLLOUT}, batch=${ROLLOUT_BATCH_SIZE}, n=${N_SAMPLES_PER_PROMPT}, max_len=${ROLLOUT_MAX_RESPONSE_LEN}, temp=${ROLLOUT_TEMPERATURE}, top_p=${ROLLOUT_TOP_P}, top_k=${ROLLOUT_TOP_K}"
echo "  eval: interval=${EVAL_INTERVAL}, n=${N_SAMPLES_PER_EVAL_PROMPT}, max_len=${EVAL_MAX_RESPONSE_LEN}, temp=${EVAL_TEMPERATURE}, top_p=${EVAL_TOP_P}, top_k=${EVAL_TOP_K}"
echo "  policy loss: ${POLICY_LOSS_TYPE}"
echo "  compression OPD: ${USE_COMPRESSION_OPD}"
echo "  swanlab: ${USE_SWANLAB}"

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

EXTRA_TRACKING_ARGS=()
if [[ "${USE_SWANLAB}" == "true" ]]; then
  EXTRA_TRACKING_ARGS+=(--use-swanlab)
  EXTRA_TRACKING_ARGS+=(--swanlab-mode "${SWANLAB_MODE}")
  EXTRA_TRACKING_ARGS+=(--swanlab-project "${SWANLAB_PROJECT}")
  EXTRA_TRACKING_ARGS+=(--swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME}")
  EXTRA_TRACKING_ARGS+=(--swanlab-group "${SWANLAB_GROUP}")
  EXTRA_TRACKING_ARGS+=(--swanlab-tags "${SWANLAB_TAGS}")
  if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
    EXTRA_TRACKING_ARGS+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
  fi
  if [[ -n "${SWANLAB_LOGDIR}" ]]; then
    EXTRA_TRACKING_ARGS+=(--swanlab-logdir "${SWANLAB_LOGDIR}")
  fi
  if [[ -n "${SWANLAB_API_KEY}" ]]; then
    EXTRA_TRACKING_ARGS+=(--swanlab-api-key "${SWANLAB_API_KEY}")
  fi
  if [[ -n "${SWANLAB_HOST}" ]]; then
    EXTRA_TRACKING_ARGS+=(--swanlab-host "${SWANLAB_HOST}")
  fi
  if [[ -n "${SWANLAB_RUN_ID}" ]]; then
    EXTRA_TRACKING_ARGS+=(--swanlab-run-id "${SWANLAB_RUN_ID}")
  fi
  if [[ -n "${SWANLAB_RESUME}" ]]; then
    EXTRA_TRACKING_ARGS+=(--swanlab-resume "${SWANLAB_RESUME}")
  fi
  if [[ -n "${SWANLAB_PUBLIC}" ]]; then
    if [[ "${SWANLAB_PUBLIC}" == "true" ]]; then
      EXTRA_TRACKING_ARGS+=(--swanlab-public)
    else
      EXTRA_TRACKING_ARGS+=(--no-swanlab-public)
    fi
  fi
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
    --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE}" \
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
  --save-interval "${SAVE_INTERVAL}" \
  --prompt-data "${TRAIN_DATA}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
  --rollout-temperature "${ROLLOUT_TEMPERATURE}" \
  --rollout-top-p "${ROLLOUT_TOP_P}" \
  --rollout-top-k "${ROLLOUT_TOP_K}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --balance-data \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-prompt-data aime24 "${AIME24}" aime25 "${AIME25}" math500 "${MATH500}" \
  --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}" \
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}" \
  --eval-temperature "${EVAL_TEMPERATURE}" \
  --eval-top-p "${EVAL_TOP_P}" \
  --eval-top-k "${EVAL_TOP_K}" \
  --tensor-model-parallel-size "${STUDENT_TP}" \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --advantage-estimator "${ADVANTAGE_ESTIMATOR}" \
  --use-opd \
  --opd-type sglang \
  --opd-kl-coef "${OPD_KL_COEF}" \
  --policy-loss-type "${POLICY_LOSS_TYPE}" \
  --future-kl-decay-rate "${FUTURE_KL_DECAY_RATE}" \
  --future-kl-start "${FUTURE_KL_START}" \
  --future-kl-window "${FUTURE_KL_WINDOW}" \
  --future-kl-clip-ratio "${FUTURE_KL_CLIP_RATIO}" \
  --future-kl-safety-threshold "${FUTURE_KL_SAFETY_THRESHOLD}" \
  --eps-clip-c "${EPS_CLIP_C}" \
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
  --optimizer "${OPTIMIZER}" \
  --lr "${LR}" \
  --lr-decay-style "${LR_DECAY_STYLE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --adam-beta1 "${ADAM_BETA1}" \
  --adam-beta2 "${ADAM_BETA2}" \
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}" \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func \
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
  "${EXTRA_TRACKING_ARGS[@]}" \
  --rm-url "${TEACHER_URL}"

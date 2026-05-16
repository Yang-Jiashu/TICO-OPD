#!/usr/bin/env bash
set -euo pipefail

# Run from the slime repo root after applying the TICO-OPD patch.
# This script is intentionally conservative and mirrors slime's Qwen3-4B layout.

SLIME_DIR=${SLIME_DIR:-/root/slime}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
BASE_MODEL=${BASE_MODEL:-/root/Qwen3-4B}
TEACHER_MODEL=${TEACHER_MODEL:-/root/Qwen3-32B}
TRAIN_DATA=${TRAIN_DATA:-/root/data/math/dapo17k.jsonl}
AIME24=${AIME24:-/root/data/math/aime24.jsonl}
AIME25=${AIME25:-/root/data/math/aime25.jsonl}
MATH500=${MATH500:-/root/data/math/math500.jsonl}
SAVE_DIR=${SAVE_DIR:-/root/checkpoints/qwen3-4b-tico-opd}
REF_LOAD=${REF_LOAD:-/root/Qwen3-4B_torch_dist}
NUM_GPUS=${NUM_GPUS:-8}

TEACHER_IP=${TEACHER_IP:-127.0.0.1}
TEACHER_PORT=${TEACHER_PORT:-13141}
TEACHER_URL="http://${TEACHER_IP}:${TEACHER_PORT}/generate"

source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

python3 -m sglang.launch_server \
  --model-path "${TEACHER_MODEL}" \
  --host 0.0.0.0 \
  --port "${TEACHER_PORT}" \
  --tp 1 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.6 &

until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; do
  echo "waiting for teacher at ${TEACHER_URL}"
  sleep 5
done

ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
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
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 9216 \
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
  --rollout-num-gpus-per-engine 2 \
  --sglang-mem-fraction-static 0.7 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func \
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
  --rm-url "${TEACHER_URL}"

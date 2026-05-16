#!/usr/bin/env bash

# Shared Qwen3 size mapping for TICO-OPD scripts.
# Supported dense sizes: 0.6B, 1.7B, 4B, 8B, 14B, 32B.
# Supported MoE sizes: 30B-A3B, 235B-A22B.

qwen3_model_id() {
  case "$1" in
    0.6B) echo "Qwen/Qwen3-0.6B" ;;
    1.7B) echo "Qwen/Qwen3-1.7B" ;;
    4B) echo "Qwen/Qwen3-4B" ;;
    4B-Instruct-2507) echo "Qwen/Qwen3-4B-Instruct-2507" ;;
    8B) echo "Qwen/Qwen3-8B" ;;
    14B) echo "Qwen/Qwen3-14B" ;;
    32B) echo "Qwen/Qwen3-32B" ;;
    30B-A3B) echo "Qwen/Qwen3-30B-A3B" ;;
    235B-A22B) echo "Qwen/Qwen3-235B-A22B" ;;
    *) echo "Unsupported Qwen3 size: $1" >&2; return 1 ;;
  esac
}

qwen3_slime_model_args_file() {
  case "$1" in
    0.6B) echo "qwen3-0.6B.sh" ;;
    1.7B) echo "qwen3-1.7B.sh" ;;
    4B) echo "qwen3-4B.sh" ;;
    4B-Instruct-2507) echo "qwen3-4B-Instruct-2507.sh" ;;
    8B) echo "qwen3-8B.sh" ;;
    14B) echo "qwen3-14B.sh" ;;
    32B) echo "qwen3-32B.sh" ;;
    30B-A3B) echo "qwen3-30B-A3B.sh" ;;
    235B-A22B) echo "qwen3-235B-A22B.sh" ;;
    *) echo "Unsupported Qwen3 size: $1" >&2; return 1 ;;
  esac
}

qwen3_default_tp() {
  case "$1" in
    0.6B|1.7B|4B|4B-Instruct-2507|8B) echo "1" ;;
    14B|32B|30B-A3B) echo "2" ;;
    235B-A22B) echo "8" ;;
    *) echo "1" ;;
  esac
}

qwen3_default_max_tokens_per_gpu() {
  case "$1" in
    0.6B|1.7B|4B|4B-Instruct-2507) echo "9216" ;;
    8B|14B) echo "8192" ;;
    32B|30B-A3B) echo "6144" ;;
    235B-A22B) echo "4096" ;;
    *) echo "8192" ;;
  esac
}

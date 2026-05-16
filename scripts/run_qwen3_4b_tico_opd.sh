#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export STUDENT_SIZE="${STUDENT_SIZE:-4B}"
export TEACHER_SIZE="${TEACHER_SIZE:-32B}"

exec "${SCRIPT_DIR}/run_qwen3_tico_opd.sh"

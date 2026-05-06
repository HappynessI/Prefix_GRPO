#!/bin/bash
set -eo pipefail

# ALFWorld GRPO+MIS training entrypoint. The base GRPO script owns environment
# setup and training; this wrapper only enables rollout correction defaults.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_SCRIPT="${SCRIPT_DIR}/run_alfworld_grpo_train.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
    echo "错误: 基础训练脚本不存在: ${BASE_SCRIPT}"
    exit 1
fi

OUTPUT_ROOT_DEFAULT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"

export GRPO_MIS_ENABLE=true
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT_DEFAULT}/alfworld_grpo_mis}"
export ROLLOUT_IS="${ROLLOUT_IS:-sequence}"
export ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
export ROLLOUT_RS="${ROLLOUT_RS:-sequence}"
export ROLLOUT_RS_THRESHOLD="${ROLLOUT_RS_THRESHOLD:-2.0}"
export ROLLOUT_RS_THRESHOLD_LOWER="${ROLLOUT_RS_THRESHOLD_LOWER:-0.2}"

exec bash "${BASE_SCRIPT}"

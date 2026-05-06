#!/bin/bash
set -eo pipefail

# DAPO-style TextCraft GRPO preset.
# This keeps the same effective runner defaults and only enables clip-higher:
#   clip_ratio_low=0.2
#   clip_ratio_high=0.28
#
# Note: this is not the full DAPO recipe; it is the minimal clip-higher ablation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_SCRIPT="${SCRIPT_DIR}/run_textcraft_grpo_train.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
    echo "错误: 基础训练脚本不存在: ${BASE_SCRIPT}"
    exit 1
fi

if ! grep -q 'ACTOR_CLIP_OVERRIDE_ARGS' "${BASE_SCRIPT}" || \
   ! grep -q 'actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}' "${BASE_SCRIPT}"; then
    echo "错误: 基础训练脚本不支持 clip-higher 覆盖: ${BASE_SCRIPT}"
    echo "请同步更新 run_textcraft_grpo_train.sh 后再运行该脚本。"
    exit 1
fi

OUTPUT_ROOT_DEFAULT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"

export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT_DEFAULT}/textcraft_grpo_dapo_clip_higher}"

export NUM_GPUS="${NUM_GPUS:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
export PPO_EPOCHS="${PPO_EPOCHS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"

export ROLLOUT_N="${ROLLOUT_N:-8}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ROLLOUT_RESPONSE_LENGTH="${ROLLOUT_RESPONSE_LENGTH:-8192}"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-512}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-12288}"

export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.80}"

export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-30}"
export MAX_USER_TURNS="${MAX_USER_TURNS:-30}"

export SAVE_FREQ="${SAVE_FREQ:-200}"
export TEST_FREQ="${TEST_FREQ:-200}"
export VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-false}"
export VAL_N="${VAL_N:-1}"

export CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
export CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"

exec bash "${BASE_SCRIPT}"

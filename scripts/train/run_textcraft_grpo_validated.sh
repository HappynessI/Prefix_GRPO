#!/bin/bash
set -eo pipefail

# TextCraft Prefix-GRPO 主实验脚本（Validated 数据）
# H200 版本沿用 scripts/train 的 Pod 内自举方式，并补齐开发机主实验入口的
# debug / preflight / shared-clip / prefix 参数覆盖。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_PROJECT_ROOT="${PROJECT_ROOT:-}"
PROJECT_ROOT="$SCRIPT_PROJECT_ROOT"
DEBUG_DIR="${SCRIPT_DIR}/debug"
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"${PROJECT_ROOT}/data/textcraft/replay_validated/main_change_top3_w11_fullflow.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/textcraft_grpo_validated"}
INTERACTION_CONFIG="${PROJECT_ROOT}/config/interaction_config/textcraft_interaction.yaml"
VERL_ROOT="${PROJECT_ROOT}/verl"
CONFIG_ROOT="${PROJECT_ROOT}/config"
AGENTGYM_ROOT="${PROJECT_ROOT}/envs/AgentGym"
WHEEL_DIR="${PROJECT_ROOT}/third_party/wheels_py312"
RUNTIME_REQS="${PROJECT_ROOT}/third_party/requirements_textcraft_runtime.txt"
VERL_WHEEL_DIR="${PROJECT_ROOT}/third_party/wheels_verl_py312"
VERL_RUNTIME_REQS="${PROJECT_ROOT}/third_party/requirements_verl_runtime.txt"
TEXTCRAFT_PORT=${TEXTCRAFT_PORT:-36001}
TEXTCRAFT_SERVER_COUNT=${TEXTCRAFT_SERVER_COUNT:-1}
TEXTCRAFT_SERVER="http://127.0.0.1:${TEXTCRAFT_PORT}"
DATA_VALIDATION_PROTOCOL="20260420_allow_raw_zero_prefix_rows"

DEBUG_MODE=${DEBUG_MODE:-0}
DEBUG_MAX_SAMPLES=${DEBUG_MAX_SAMPLES:-16}
DEBUG_PREFLIGHT_ONLY=${DEBUG_PREFLIGHT_ONLY:-0}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        --preflight)
            DEBUG_MODE=1
            DEBUG_PREFLIGHT_ONLY=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

NUM_GPUS=${NUM_GPUS:-2}
if [ -z "${GPU_IDS:-}" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

SAVE_FREQ_IS_SET=${SAVE_FREQ+x}
TEST_FREQ_IS_SET=${TEST_FREQ+x}
CLIP_RATIO_IS_SET=${CLIP_RATIO+x}
CLIP_RATIO_LOW_IS_SET=${CLIP_RATIO_LOW+x}
CLIP_RATIO_HIGH_IS_SET=${CLIP_RATIO_HIGH+x}
CLIP_RATIO_C_IS_SET=${CLIP_RATIO_C+x}
PREFIX_CLIP_RATIO_IS_SET=${PREFIX_CLIP_RATIO+x}
PREFIX_CLIP_RATIO_LOW_IS_SET=${PREFIX_CLIP_RATIO_LOW+x}
PREFIX_CLIP_RATIO_HIGH_IS_SET=${PREFIX_CLIP_RATIO_HIGH+x}
PREFIX_CLIP_RATIO_C_IS_SET=${PREFIX_CLIP_RATIO_C+x}

NUM_EPOCHS=${NUM_EPOCHS:-30}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_EPOCHS=${PPO_EPOCHS:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
SAVE_FREQ=${SAVE_FREQ:-500}
TEST_FREQ=${TEST_FREQ:-500}

CLIP_RATIO=${CLIP_RATIO:-}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-}
CLIP_RATIO_C=${CLIP_RATIO_C:-}
DIFF_LOW=${DIFF_LOW:-}
DIFF_HIGH=${DIFF_HIGH:-}

OPTIMIZE_PREFIX_TOKENS=${OPTIMIZE_PREFIX_TOKENS:-true}
PREFIX_LOSS_WEIGHT=${PREFIX_LOSS_WEIGHT:-1.0}
PREFIX_LOSS_MODE=${PREFIX_LOSS_MODE:-split}
PREFIX_ADVANTAGE_MODE=${PREFIX_ADVANTAGE_MODE:-cont_mean_abs}
PREFIX_ADVANTAGE_CONSTANT=${PREFIX_ADVANTAGE_CONSTANT:-1.0}
PREFIX_CONT_ADV_WEIGHT=${PREFIX_CONT_ADV_WEIGHT:-1.0}
PREFIX_FAMILY_LIFT_WEIGHT=${PREFIX_FAMILY_LIFT_WEIGHT:-1.0}
PREFIX_FAMILY_LIFT_CLIP=${PREFIX_FAMILY_LIFT_CLIP:-1.0}
PREFIX_CLIP_RATIO=${PREFIX_CLIP_RATIO:-0.2}
PREFIX_CLIP_RATIO_LOW=${PREFIX_CLIP_RATIO_LOW:-0.2}
PREFIX_CLIP_RATIO_HIGH=${PREFIX_CLIP_RATIO_HIGH:-0.2}
PREFIX_CLIP_RATIO_C=${PREFIX_CLIP_RATIO_C:-3.0}
PREFIX_DIFF_LOW=${PREFIX_DIFF_LOW:-}
PREFIX_DIFF_HIGH=${PREFIX_DIFF_HIGH:-}
USE_TEXTCRAFT_BC_AUX=${USE_TEXTCRAFT_BC_AUX:-false}
TEXTCRAFT_BC_WEIGHT=${TEXTCRAFT_BC_WEIGHT:-0.0}
TEXTCRAFT_BC_SOURCE=${TEXTCRAFT_BC_SOURCE:-prefix}
TEXTCRAFT_BC_MAX_LENGTH=${TEXTCRAFT_BC_MAX_LENGTH:-}
USE_TEXTCRAFT_TEACHER_DEMO=${USE_TEXTCRAFT_TEACHER_DEMO:-false}
TEXTCRAFT_TEACHER_DEMO_WEIGHT=${TEXTCRAFT_TEACHER_DEMO_WEIGHT:-1.0}
TEXTCRAFT_TEACHER_DEMO_LABELS=${TEXTCRAFT_TEACHER_DEMO_LABELS:-teacher_demo,demo}
HYDRA_TEXTCRAFT_TEACHER_DEMO_LABELS="'${TEXTCRAFT_TEACHER_DEMO_LABELS}'"
TEXTCRAFT_TEACHER_DEMO_REPEAT_TO_ROLLOUT_N=${TEXTCRAFT_TEACHER_DEMO_REPEAT_TO_ROLLOUT_N:-true}
TEXTCRAFT_TEACHER_DEMO_SKIP_OVERLONG=${TEXTCRAFT_TEACHER_DEMO_SKIP_OVERLONG:-false}
USE_KL_LOSS=${USE_KL_LOSS:-false}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-false}
ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-false}

ROLLOUT_N=${ROLLOUT_N:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}

VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}
VAL_N=${VAL_N:-1}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-8192}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-512}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-12288}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-30}
MAX_USER_TURNS=${MAX_USER_TURNS:-30}

RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
DATA_SHUFFLE_WAS_SET=${DATA_SHUFFLE+x}
DATA_SHUFFLE=${DATA_SHUFFLE:-true}
METRICS_CSV_FREQ=${METRICS_CSV_FREQ:-50}
METRICS_CSV_FILENAME=${METRICS_CSV_FILENAME:-training_metrics.csv}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

normalize_prefix_advantage_mode() {
    local raw_mode="$PREFIX_ADVANTAGE_MODE"
    local normalized_mode="${raw_mode,,}"
    normalized_mode="${normalized_mode//$'\r'/}"
    normalized_mode="${normalized_mode//$'\n'/}"
    normalized_mode="${normalized_mode//$'\t'/}"
    normalized_mode="${normalized_mode// /}"

    case "$normalized_mode" in
        cont_mean|cont_mean_abs|cont_mean_pos|constant|success_gate_constant|family_lift|family_lift_pos|cont_abs_plus_family_lift|cont_abs_plus_family_lift_pos|family_lift_pos_cont_abs)
            PREFIX_ADVANTAGE_MODE="$normalized_mode"
            ;;
        family_lift_positive)
            PREFIX_ADVANTAGE_MODE="family_lift_pos"
            ;;
        cont_abs_family_lift)
            PREFIX_ADVANTAGE_MODE="cont_abs_plus_family_lift"
            ;;
        cont_abs_family_lift_pos)
            PREFIX_ADVANTAGE_MODE="cont_abs_plus_family_lift_pos"
            ;;
        contstant)
            echo "警告: 检测到 PREFIX_ADVANTAGE_MODE=$raw_mode，自动更正为 constant" | tee -a "$LOG_FILE" >&2
            PREFIX_ADVANTAGE_MODE="constant"
            ;;
        *)
            echo "错误: 不支持的 PREFIX_ADVANTAGE_MODE=$raw_mode，可选值: cont_mean / cont_mean_abs / cont_mean_pos / constant / success_gate_constant / family_lift / family_lift_pos / cont_abs_plus_family_lift / cont_abs_plus_family_lift_pos" | tee -a "$LOG_FILE" >&2
            exit 1
            ;;
    esac
}

convert_log_ratio_diff_to_clip_delta() {
    local direction="$1"
    local raw_value="$2"

    python3 - "$direction" "$raw_value" <<'PY'
import math
import sys

direction, raw_value = sys.argv[1], sys.argv[2]
value = float(raw_value)
if value < 0:
    raise ValueError(f"expected non-negative diff value, got {value}")
if direction == "low":
    result = 1.0 - math.exp(-value)
elif direction == "high":
    result = math.exp(value) - 1.0
else:
    raise ValueError(f"unsupported direction: {direction}")
print(f"{result:.12g}")
PY
}

apply_diff_alias_override() {
    local alias_name="$1"
    local target_name="$2"
    local direction="$3"
    local target_was_set="$4"
    local alias_value="${!alias_name:-}"

    if [ -z "$alias_value" ]; then
        return 0
    fi

    if [ -n "$target_was_set" ]; then
        echo "警告: ${alias_name}=${alias_value} 已提供，但 ${target_name} 也已显式设置；忽略 ${alias_name}。" | tee -a "$LOG_FILE" >&2
        return 0
    fi

    local converted_value
    if ! converted_value="$(convert_log_ratio_diff_to_clip_delta "$direction" "$alias_value" 2>>"$LOG_FILE")"; then
        echo "错误: ${alias_name}=${alias_value} 非法，必须是 >= 0 的数值。" | tee -a "$LOG_FILE" >&2
        exit 1
    fi

    printf -v "$target_name" '%s' "$converted_value"
    echo "应用 ${alias_name}=${alias_value} -> ${target_name}=${converted_value}" | tee -a "$LOG_FILE"
}

resolve_effective_clip_config() {
    apply_diff_alias_override DIFF_LOW CLIP_RATIO_LOW low "$CLIP_RATIO_LOW_IS_SET"
    apply_diff_alias_override DIFF_HIGH CLIP_RATIO_HIGH high "$CLIP_RATIO_HIGH_IS_SET"
    apply_diff_alias_override PREFIX_DIFF_LOW PREFIX_CLIP_RATIO_LOW low "$PREFIX_CLIP_RATIO_LOW_IS_SET"
    apply_diff_alias_override PREFIX_DIFF_HIGH PREFIX_CLIP_RATIO_HIGH high "$PREFIX_CLIP_RATIO_HIGH_IS_SET"

    EFFECTIVE_CLIP_RATIO=${CLIP_RATIO:-0.2}
    EFFECTIVE_CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-$EFFECTIVE_CLIP_RATIO}
    EFFECTIVE_CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-$EFFECTIVE_CLIP_RATIO}
    EFFECTIVE_CLIP_RATIO_C=${CLIP_RATIO_C:-3.0}

    EFFECTIVE_PREFIX_CLIP_RATIO=${PREFIX_CLIP_RATIO:-$EFFECTIVE_CLIP_RATIO}
    EFFECTIVE_PREFIX_CLIP_RATIO_LOW=${PREFIX_CLIP_RATIO_LOW:-$EFFECTIVE_CLIP_RATIO_LOW}
    EFFECTIVE_PREFIX_CLIP_RATIO_HIGH=${PREFIX_CLIP_RATIO_HIGH:-$EFFECTIVE_CLIP_RATIO_HIGH}
    EFFECTIVE_PREFIX_CLIP_RATIO_C=${PREFIX_CLIP_RATIO_C:-$EFFECTIVE_CLIP_RATIO_C}
}

if [ "$DEBUG_MODE" = "1" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_debug"
    NUM_EPOCHS=1
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=4
    MICRO_BATCH_SIZE=2
    ROLLOUT_N=2
    if [ -z "$SAVE_FREQ_IS_SET" ]; then
        SAVE_FREQ=1
    fi
    if [ -z "$TEST_FREQ_IS_SET" ]; then
        TEST_FREQ=1
    fi
    GPU_MEMORY_UTIL=0.5
    MAX_NUM_SEQS=16
    MAX_RESPONSE_LENGTH=512
    ROLLOUT_RESPONSE_LENGTH=512
    MAX_NUM_BATCHED_TOKENS=4096
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="textcraft_grpo_validated_${TIMESTAMP}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"
TEXTCRAFT_LOG="$LOG_DIR/textcraft_env_${TIMESTAMP}.log"

normalize_prefix_advantage_mode
case "$PREFIX_ADVANTAGE_MODE" in
    family_lift|family_lift_pos|cont_abs_plus_family_lift|cont_abs_plus_family_lift_pos|family_lift_pos_cont_abs)
        if [ -z "$DATA_SHUFFLE_WAS_SET" ]; then
            DATA_SHUFFLE=false
        fi
        ;;
esac
if [ "$USE_TEXTCRAFT_TEACHER_DEMO" = "true" ] && [ -z "$DATA_SHUFFLE_WAS_SET" ]; then
    DATA_SHUFFLE=false
fi
resolve_effective_clip_config
TEXTCRAFT_BC_MAX_LENGTH_OVERRIDE=${TEXTCRAFT_BC_MAX_LENGTH:-null}

GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    echo "错误: GPU_IDS中的GPU数量($GPU_COUNT)与NUM_GPUS($NUM_GPUS)不一致！" | tee -a "$LOG_FILE"
    exit 1
fi
if ! [[ "$TEXTCRAFT_SERVER_COUNT" =~ ^[0-9]+$ ]] || [ "$TEXTCRAFT_SERVER_COUNT" -lt 1 ]; then
    echo "错误: TEXTCRAFT_SERVER_COUNT=$TEXTCRAFT_SERVER_COUNT 非法，必须是 >= 1 的整数。" | tee -a "$LOG_FILE"
    exit 1
fi

cleanup_textcraft() {
    local cleaned=0
    for pid in "${TEXTCRAFT_PIDS[@]:-}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            if [ "$cleaned" -eq 0 ]; then
                echo "" | tee -a "$LOG_FILE"
                echo "清理 TextCraft 环境服务..." | tee -a "$LOG_FILE"
            fi
            cleaned=1
            echo "  kill TextCraft PID=$pid" | tee -a "$LOG_FILE"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [ "$cleaned" -eq 1 ]; then
        echo "TextCraft 环境服务已清理。" | tee -a "$LOG_FILE"
    fi
}
trap cleanup_textcraft EXIT

echo "============================================" | tee -a "$LOG_FILE"
echo "TextCraft Prefix-GRPO 主实验 (Validated)" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "data_validation_protocol: $DATA_VALIDATION_PROTOCOL" | tee -a "$LOG_FILE"
if [ -n "$ENV_PROJECT_ROOT" ] && [ "$ENV_PROJECT_ROOT" != "$PROJECT_ROOT" ]; then
    echo "警告: 外部 PROJECT_ROOT=$ENV_PROJECT_ROOT 与脚本所在根目录 $PROJECT_ROOT 不一致；代码/依赖路径以脚本所在根目录为准。" | tee -a "$LOG_FILE"
fi
echo "项目根目录: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "TextCraft wheel目录: $WHEEL_DIR" | tee -a "$LOG_FILE"
echo "verl wheel目录: $VERL_WHEEL_DIR" | tee -a "$LOG_FILE"
echo "TextCraft server_count: $TEXTCRAFT_SERVER_COUNT" | tee -a "$LOG_FILE"
echo "模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "训练数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "gradient_checkpointing: $ENABLE_GRADIENT_CHECKPOINTING" | tee -a "$LOG_FILE"
echo "activation_offload: $ENABLE_ACTIVATION_OFFLOAD" | tee -a "$LOG_FILE"
echo "clip_ratio: $EFFECTIVE_CLIP_RATIO" | tee -a "$LOG_FILE"
echo "clip_ratio_low: $EFFECTIVE_CLIP_RATIO_LOW" | tee -a "$LOG_FILE"
echo "clip_ratio_high: $EFFECTIVE_CLIP_RATIO_HIGH" | tee -a "$LOG_FILE"
echo "clip_ratio_c: $EFFECTIVE_CLIP_RATIO_C" | tee -a "$LOG_FILE"
echo "optimize_prefix_tokens: $OPTIMIZE_PREFIX_TOKENS" | tee -a "$LOG_FILE"
echo "prefix_loss_weight: $PREFIX_LOSS_WEIGHT" | tee -a "$LOG_FILE"
echo "prefix_loss_mode: $PREFIX_LOSS_MODE" | tee -a "$LOG_FILE"
echo "prefix_advantage_mode: $PREFIX_ADVANTAGE_MODE" | tee -a "$LOG_FILE"
echo "prefix_advantage_constant: $PREFIX_ADVANTAGE_CONSTANT" | tee -a "$LOG_FILE"
echo "prefix_cont_adv_weight: $PREFIX_CONT_ADV_WEIGHT" | tee -a "$LOG_FILE"
echo "prefix_family_lift_weight: $PREFIX_FAMILY_LIFT_WEIGHT" | tee -a "$LOG_FILE"
echo "prefix_family_lift_clip: $PREFIX_FAMILY_LIFT_CLIP" | tee -a "$LOG_FILE"
echo "prefix_clip_ratio: $EFFECTIVE_PREFIX_CLIP_RATIO" | tee -a "$LOG_FILE"
echo "prefix_clip_ratio_low: $EFFECTIVE_PREFIX_CLIP_RATIO_LOW" | tee -a "$LOG_FILE"
echo "prefix_clip_ratio_high: $EFFECTIVE_PREFIX_CLIP_RATIO_HIGH" | tee -a "$LOG_FILE"
echo "prefix_clip_ratio_c: $EFFECTIVE_PREFIX_CLIP_RATIO_C" | tee -a "$LOG_FILE"
echo "diff_low: ${DIFF_LOW:-<unset>}" | tee -a "$LOG_FILE"
echo "diff_high: ${DIFF_HIGH:-<unset>}" | tee -a "$LOG_FILE"
echo "prefix_diff_low: ${PREFIX_DIFF_LOW:-<unset>}" | tee -a "$LOG_FILE"
echo "prefix_diff_high: ${PREFIX_DIFF_HIGH:-<unset>}" | tee -a "$LOG_FILE"
echo "use_textcraft_bc_aux: $USE_TEXTCRAFT_BC_AUX" | tee -a "$LOG_FILE"
echo "textcraft_bc_weight: $TEXTCRAFT_BC_WEIGHT" | tee -a "$LOG_FILE"
echo "textcraft_bc_source: $TEXTCRAFT_BC_SOURCE" | tee -a "$LOG_FILE"
echo "textcraft_bc_max_length: ${TEXTCRAFT_BC_MAX_LENGTH:-<auto>}" | tee -a "$LOG_FILE"
echo "use_textcraft_teacher_demo: $USE_TEXTCRAFT_TEACHER_DEMO" | tee -a "$LOG_FILE"
echo "textcraft_teacher_demo_weight: $TEXTCRAFT_TEACHER_DEMO_WEIGHT" | tee -a "$LOG_FILE"
echo "textcraft_teacher_demo_labels: $TEXTCRAFT_TEACHER_DEMO_LABELS" | tee -a "$LOG_FILE"
echo "textcraft_teacher_demo_repeat_to_rollout_n: $TEXTCRAFT_TEACHER_DEMO_REPEAT_TO_ROLLOUT_N" | tee -a "$LOG_FILE"
echo "textcraft_teacher_demo_skip_overlong: $TEXTCRAFT_TEACHER_DEMO_SKIP_OVERLONG" | tee -a "$LOG_FILE"
echo "data.shuffle: $DATA_SHUFFLE" | tee -a "$LOG_FILE"
echo "use_kl_loss: $USE_KL_LOSS" | tee -a "$LOG_FILE"
if [ "$DEBUG_MODE" = "1" ]; then
    echo "DEBUG_MODE=1, DEBUG_MAX_SAMPLES=$DEBUG_MAX_SAMPLES, DEBUG_PREFLIGHT_ONLY=$DEBUG_PREFLIGHT_ONLY" | tee -a "$LOG_FILE"
fi
echo "============================================" | tee -a "$LOG_FILE"

echo "离线安装 TextCraft 运行时依赖..." | tee -a "$LOG_FILE"
if [ ! -d "$WHEEL_DIR" ]; then
    echo "错误: wheel 目录不存在: $WHEEL_DIR" | tee -a "$LOG_FILE"
    exit 1
fi
if ! python3 -m pip install --no-index --find-links="$WHEEL_DIR" pip setuptools wheel 2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: pip/setuptools/wheel 离线安装失败。" | tee -a "$LOG_FILE"
    exit 1
fi
if ! python3 -m pip install --no-index --find-links="$WHEEL_DIR" -r "$RUNTIME_REQS" 2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: TextCraft 运行时依赖离线安装失败。" | tee -a "$LOG_FILE"
    exit 1
fi

echo "安装 agentenv-textcraft..." | tee -a "$LOG_FILE"
cd "$AGENTGYM_ROOT/agentenv-textcraft"
if ! python3 -m pip install --no-build-isolation -e . --no-deps -q 2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: agentenv-textcraft 安装失败" | tee -a "$LOG_FILE"
    exit 1
fi

echo "启动 TextCraft 服务池 (base_port=${TEXTCRAFT_PORT}, count=${TEXTCRAFT_SERVER_COUNT})..." | tee -a "$LOG_FILE"
TEXTCRAFT_PIDS=()
TEXTCRAFT_SERVERS=()
TEXTCRAFT_LOGS=()
for server_idx in $(seq 0 $((TEXTCRAFT_SERVER_COUNT - 1))); do
    port=$((TEXTCRAFT_PORT + server_idx))
    server_url="http://127.0.0.1:${port}"
    server_log="$LOG_DIR/textcraft_env_${port}_${TIMESTAMP}.log"
    TEXTCRAFT_SERVERS+=("$server_url")
    TEXTCRAFT_LOGS+=("$server_log")
    echo "  启动 TextCraft[$server_idx]: $server_url log=$server_log" | tee -a "$LOG_FILE"
    textcraft --host 0.0.0.0 --port "$port" > "$server_log" 2>&1 &
    TEXTCRAFT_PIDS+=("$!")
done
TEXTCRAFT_SERVER="${TEXTCRAFT_SERVERS[0]}"
TEXTCRAFT_SERVERS_CSV="$(IFS=,; echo "${TEXTCRAFT_SERVERS[*]}")"
TEXTCRAFT_LOG="${TEXTCRAFT_LOGS[0]}"

for server_idx in "${!TEXTCRAFT_SERVERS[@]}"; do
    server_url="${TEXTCRAFT_SERVERS[$server_idx]}"
    server_log="${TEXTCRAFT_LOGS[$server_idx]}"
    READY=false
    for i in $(seq 1 60); do
        if curl -sf "${server_url}/" > /dev/null 2>&1; then
            READY=true
            break
        fi
        echo "  等待 TextCraft[$server_idx] 服务启动... ($i/60) $server_url" | tee -a "$LOG_FILE"
        sleep 2
    done
    if [ "$READY" != "true" ]; then
        echo "错误: TextCraft[$server_idx] 环境服务启动超时，请查看 $server_log" | tee -a "$LOG_FILE"
        exit 1
    fi
done
echo "TextCraft servers: $TEXTCRAFT_SERVERS_CSV" | tee -a "$LOG_FILE"

RUNTIME_INTERACTION_CONFIG="$LOG_DIR/textcraft_interaction_${TEXTCRAFT_PORT}_${TIMESTAMP}.yaml"
cat > "$RUNTIME_INTERACTION_CONFIG" <<EOF
# Runtime-generated from run_textcraft_grpo_validated.sh.
# Keep env_server_base aligned with TEXTCRAFT_PORT for this run.
interaction:
  - name: textcraft
    class_name: verl.interactions.textcraft_interaction.TextCraftInteraction
    config:
      env_server_base: "$TEXTCRAFT_SERVER"
      env_server_bases:
$(for server_url in "${TEXTCRAFT_SERVERS[@]}"; do printf '        - "%s"\n' "$server_url"; done)
      timeout: 600
      max_retries: 5
      retry_backoff: 0.5
      retry_status_codes: [408, 429, 502, 503, 504]
EOF
INTERACTION_CONFIG="$RUNTIME_INTERACTION_CONFIG"
echo "运行时 interaction config: $INTERACTION_CONFIG" | tee -a "$LOG_FILE"

echo "安装 verl 框架..." | tee -a "$LOG_FILE"
cd "$VERL_ROOT"
if [ ! -d "$VERL_WHEEL_DIR" ]; then
    echo "错误: verl wheel 目录不存在: $VERL_WHEEL_DIR" | tee -a "$LOG_FILE"
    exit 1
fi
if ! python3 -m pip install --no-index --find-links="$VERL_WHEEL_DIR" -r "$VERL_RUNTIME_REQS" 2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: verl 运行时依赖离线安装失败。" | tee -a "$LOG_FILE"
    exit 1
fi
if ! python3 -m pip install --no-build-isolation -e . --no-deps -q 2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: verl 安装失败" | tee -a "$LOG_FILE"
    exit 1
fi

VERL_CONFIG_ROOT="${VERL_ROOT}/verl/trainer/config"
if [ ! -d "$VERL_CONFIG_ROOT" ]; then
    echo "错误: verl 配置目录不存在: $VERL_CONFIG_ROOT" | tee -a "$LOG_FILE"
    exit 1
fi

if [ "$DEBUG_MODE" = "1" ]; then
    DEBUG_DATA_DIR="$OUTPUT_DIR/debug_data"
    mkdir -p "$DEBUG_DATA_DIR"
    DEBUG_DATA_PATH="$DEBUG_DATA_DIR/debug_subset_${DEBUG_MAX_SAMPLES}.parquet"
    echo "生成 debug 数据子集: $DEBUG_DATA_PATH" | tee -a "$LOG_FILE"
    python3 "$DEBUG_DIR/debug_data_utils.py" train \
        --input "$DATA_PATH" \
        --output "$DEBUG_DATA_PATH" \
        --max-samples "$DEBUG_MAX_SAMPLES" \
        2>&1 | tee -a "$LOG_FILE"
    DATA_PATH="$DEBUG_DATA_PATH"
    export VERL_LOGGING_LEVEL=DEBUG
fi

echo "校验训练数据..." | tee -a "$LOG_FILE"
export DATA_PATH
export OPTIMIZE_PREFIX_TOKENS
export USE_TEXTCRAFT_TEACHER_DEMO
export TEXTCRAFT_TEACHER_DEMO_LABELS
python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os
from collections import Counter

import pandas as pd

data_path = os.environ["DATA_PATH"]
optimize_prefix_tokens = os.environ["OPTIMIZE_PREFIX_TOKENS"].lower() == "true"
use_textcraft_teacher_demo = os.environ["USE_TEXTCRAFT_TEACHER_DEMO"].lower() == "true"
teacher_demo_labels = {
    item.strip().lower()
    for item in os.environ.get("TEXTCRAFT_TEACHER_DEMO_LABELS", "teacher_demo,demo").split(",")
    if item.strip()
}

if not os.path.exists(data_path):
    raise FileNotFoundError(f"训练数据不存在: {data_path}")

df = pd.read_parquet(data_path)
columns = list(df.columns)
required_columns = ["data_source", "prompt", "reward_model", "extra_info"]
required_prefix_columns = ["assistant_prefix_old_log_probs", "prefix_token_count", "prefix_mask", "assistant_prefix_span"]
required_teacher_demo_columns = [
    "teacher_demo_response_ids",
    "teacher_demo_response_attention_mask",
    "teacher_demo_response_loss_mask",
    "teacher_demo_old_log_probs",
    "teacher_demo_reward",
]
required_all = required_columns + (required_prefix_columns if optimize_prefix_tokens else [])
missing = [col for col in required_all if col not in columns]
if missing:
    raise ValueError(f"缺失主实验必需列: {missing}. data_path={data_path}")
if use_textcraft_teacher_demo:
    missing_demo_cols = [col for col in required_teacher_demo_columns if col not in columns]
    if missing_demo_cols:
        raise ValueError(f"启用 teacher demo 但缺失列: {missing_demo_cols}. data_path={data_path}")

if len(df) == 0:
    raise ValueError(f"训练数据为空: {data_path}")


def parse_prefix_span(span):
    if isinstance(span, dict):
        start = span.get("start")
        end = span.get("end")
    elif isinstance(span, (list, tuple)) and len(span) == 2:
        start, end = span
    else:
        raise ValueError(f"无法解析 assistant_prefix_span: {span!r}")
    start = int(start)
    end = int(end)
    if start < 0 or end < start:
        raise ValueError(f"assistant_prefix_span 非法: [{start}, {end})")
    return start, end


def optional_int(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


bad_prefix_action_rows = []
bad_protocol_rows = []
strategy_counter = Counter()
replay_counter = Counter()
cut_turn_counter = Counter()
prefix_action_len_counter = Counter()
zero_prefix_rows = 0
teacher_demo_rows = 0
bad_teacher_demo_rows = []

for idx, row in df.iterrows():
    row_dict = row.to_dict()
    extra_info = row_dict.get("extra_info", {})
    if not isinstance(extra_info, dict):
        bad_prefix_action_rows.append((idx, "extra_info_not_dict"))
        continue

    interaction_kwargs = extra_info.get("interaction_kwargs", {})
    if not isinstance(interaction_kwargs, dict):
        bad_prefix_action_rows.append((idx, "interaction_kwargs_not_dict"))
        continue

    strategy = row_dict.get("strategy", extra_info.get("strategy", "unknown"))
    replay_category = row_dict.get("replay_category", extra_info.get("replay_category", "unknown"))
    variant_label = str(row_dict.get("variant_label", extra_info.get("variant_label", ""))).lower()
    cut_turn_idx = row_dict.get("cut_turn_idx", extra_info.get("cut_turn_idx", "unknown"))
    strategy_counter[str(strategy)] += 1
    replay_counter[str(replay_category)] += 1
    cut_turn_counter[str(cut_turn_idx)] += 1

    prefix_token_count = optional_int(row_dict.get("prefix_token_count")) if "prefix_token_count" in row_dict else None
    if prefix_token_count == 0:
        zero_prefix_rows += 1

    if optimize_prefix_tokens:
        old_log_probs = row_dict["assistant_prefix_old_log_probs"]
        prefix_mask = row_dict["prefix_mask"]
        if prefix_token_count is None:
            raise ValueError(f"prefix_token_count 缺失或非法: row={idx}")
        span_start, span_end = parse_prefix_span(row_dict["assistant_prefix_span"])
        span_len = span_end - span_start

        old_len = len(old_log_probs)
        mask_len = len(prefix_mask)
        mask_sum = int(sum(int(x) for x in prefix_mask))
        if old_len != span_len or mask_len != span_len or mask_sum != prefix_token_count:
            sample_uid = row_dict.get("sample_uid", f"row_{idx}")
            bad_protocol_rows.append(
                (
                    sample_uid,
                    old_len,
                    mask_len,
                    span_len,
                    mask_sum,
                    prefix_token_count,
                )
            )

    prefix_actions = interaction_kwargs.get("prefix_actions", None)
    prefix_actions_len = len(prefix_actions) if prefix_actions is not None and hasattr(prefix_actions, "__len__") else 0
    empty_prefix_row = prefix_token_count == 0
    if prefix_token_count is None:
        empty_prefix_row = str(strategy).lower() == "raw" and str(replay_category).lower() == "raw"
    is_teacher_demo_row = variant_label in teacher_demo_labels
    if is_teacher_demo_row:
        teacher_demo_rows += 1
        try:
            response_ids = row_dict["teacher_demo_response_ids"]
            response_attention = row_dict["teacher_demo_response_attention_mask"]
            response_loss_mask = row_dict["teacher_demo_response_loss_mask"]
            old_log_probs = row_dict["teacher_demo_old_log_probs"]
            demo_reward = float(row_dict["teacher_demo_reward"])
            lengths = {len(response_ids), len(response_attention), len(response_loss_mask), len(old_log_probs)}
            if len(lengths) != 1:
                bad_teacher_demo_rows.append((idx, "length_mismatch"))
            elif len(response_ids) == 0:
                bad_teacher_demo_rows.append((idx, "empty_response"))
            elif int(sum(float(x) > 0 for x in response_attention)) != len(response_ids):
                bad_teacher_demo_rows.append((idx, "non_unit_attention_or_padding_in_sidecar"))
            elif float(sum(float(x) for x in response_loss_mask)) <= 0:
                bad_teacher_demo_rows.append((idx, "no_assistant_loss_tokens"))
            elif demo_reward < 0:
                bad_teacher_demo_rows.append((idx, "negative_demo_reward"))
        except Exception as exc:
            bad_teacher_demo_rows.append((idx, f"invalid_demo_sidecar:{exc}"))

    if empty_prefix_row:
        if prefix_actions_len != 0:
            bad_prefix_action_rows.append((idx, "raw_zero_prefix_actions_not_empty"))
    else:
        if prefix_actions_len == 0:
            bad_prefix_action_rows.append((idx, "prefix_actions_empty"))
        else:
            prefix_action_len_counter[prefix_actions_len] += 1

prefix_token_count = df["prefix_token_count"] if "prefix_token_count" in df.columns else None
if prefix_token_count is not None and (prefix_token_count < 0).any():
    bad = int((prefix_token_count < 0).sum())
    raise ValueError(f"prefix_token_count 中存在负样本: {bad}")

if bad_prefix_action_rows:
    preview = bad_prefix_action_rows[:5]
    raise ValueError(f"prefix_actions 协议非法的样本存在，前5条={preview}")

if bad_protocol_rows:
    preview = bad_protocol_rows[:5]
    raise ValueError(
        "prefix sidecar 协议校验失败。前5条="
        f"{preview}. 期望 len(old_log_probs)==len(prefix_mask)==span_len 且 sum(prefix_mask)==prefix_token_count"
    )
if use_textcraft_teacher_demo:
    if teacher_demo_rows == 0:
        raise ValueError("USE_TEXTCRAFT_TEACHER_DEMO=true 但数据中没有 teacher_demo/demo 行。")
    if bad_teacher_demo_rows:
        preview = bad_teacher_demo_rows[:5]
        raise ValueError(f"teacher demo sidecar 协议非法，前5条={preview}")

mode = "prefix optimization" if optimize_prefix_tokens else "baseline"
print(f"  模式: {mode}")
print(f"  样本数: {len(df)}")
if prefix_token_count is not None:
    print(f"  prefix_token_count: min={prefix_token_count.min()}, max={prefix_token_count.max()}, mean={prefix_token_count.mean():.1f}")
    print(f"  zero_prefix_rows: {zero_prefix_rows}")
print(f"  strategy分布: {dict(strategy_counter)}")
print(f"  replay_category分布: {dict(replay_counter)}")
print(f"  cut_turn_idx分布(前10项): {dict(list(cut_turn_counter.items())[:10])}")
print(f"  prefix_actions长度分布: {dict(prefix_action_len_counter)}")
if use_textcraft_teacher_demo:
    print(f"  teacher_demo_rows: {teacher_demo_rows}")
print("  数据校验通过")
PY

echo "校验 TextCraft /create 端到端可用性..." | tee -a "$LOG_FILE"
export TEXTCRAFT_SERVER
export TEXTCRAFT_SERVERS_CSV
python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os
import re

import pandas as pd
import requests

data_path = os.environ["DATA_PATH"]
server = os.environ["TEXTCRAFT_SERVER"].rstrip("/")
servers = [item.strip().rstrip("/") for item in os.environ.get("TEXTCRAFT_SERVERS_CSV", server).split(",") if item.strip()]


def extract_from_prompt(prompt):
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        return None, None
    for msg in prompt:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        goal_match = re.search(r"Goal:\s*craft\s+(.+?)\.?$", content, re.IGNORECASE | re.MULTILINE)
        commands_match = re.search(
            r"Crafting commands:\n(.+?)\n\nGoal:\s*craft\s+.+?\.?$",
            content,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if goal_match:
            commands = commands_match.group(1).strip() if commands_match else None
            return goal_match.group(1).strip(), commands
    return None, None


df = pd.read_parquet(data_path)
row = df.iloc[0].to_dict()
extra_info = row.get("extra_info", {})
interaction_kwargs = extra_info.get("interaction_kwargs", {}) if isinstance(extra_info, dict) else {}
if not isinstance(interaction_kwargs, dict):
    interaction_kwargs = {}

goal = interaction_kwargs.get("goal")
commands = interaction_kwargs.get("commands")
if goal is None or commands is None:
    parsed_goal, parsed_commands = extract_from_prompt(row.get("prompt"))
    goal = goal if goal is not None else parsed_goal
    commands = commands if commands is not None else parsed_commands

body = {}
if goal is not None:
    body["goal"] = goal
if commands:
    body["commands"] = commands
raw_data_idx = interaction_kwargs.get("data_idx", row.get("task_id", row.get("session_id", 0)))
try:
    body["data_idx"] = int(raw_data_idx)
except (TypeError, ValueError):
    body["data_idx"] = 0

for current_server in servers:
    response = requests.post(f"{current_server}/create", json=body, timeout=30)
    try:
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"TextCraft /create preflight failed on {current_server}: "
            f"status={response.status_code}, body={response.text[:500]!r}"
        ) from exc

    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"TextCraft /create returned error payload on {current_server}: {payload['error']}")
    if "id" not in payload and "env_id" not in payload:
        raise RuntimeError(f"TextCraft /create returned no env id on {current_server}: {payload}")

    env_id = payload.get("id", payload.get("env_id"))
    requests.post(f"{current_server}/close", json={"id": env_id}, timeout=10)
    print(f"  /create preflight passed: server={current_server}, env_id={env_id}, goal={goal!r}")
PY
CREATE_PREFLIGHT_EXIT_CODE=${PIPESTATUS[0]}
if [ "$CREATE_PREFLIGHT_EXIT_CODE" -ne 0 ]; then
    echo "错误: TextCraft /create 端到端校验失败，请查看 $TEXTCRAFT_LOG" | tee -a "$LOG_FILE"
    exit $CREATE_PREFLIGHT_EXIT_CODE
fi

if [ "$DEBUG_PREFLIGHT_ONLY" = "1" ]; then
    echo "运行 preflight smoke test..." | tee -a "$LOG_FILE"
    python3 "$DEBUG_DIR/preflight_test.py" \
        --project_root "$PROJECT_ROOT" \
        --data_path "$DATA_PATH" \
        --model_path "$MODEL_PATH" \
        --textcraft_server "$TEXTCRAFT_SERVER" \
        2>&1 | tee -a "$LOG_FILE"
    PRECHECK_EXIT_CODE=${PIPESTATUS[0]}
    exit $PRECHECK_EXIT_CODE
fi

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CONFIGURE_LOGGING=0
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export UV_THREADPOOL_SIZE=${UV_THREADPOOL_SIZE:-4}

cd "$VERL_ROOT"
python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_ROOT}" \
    --config-name='textcraft_grpo_train' \
    hydra.searchpath=[file://${VERL_CONFIG_ROOT},file://${CONFIG_ROOT}] \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_PATH" \
    data.val_files="$DATA_PATH" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    '+data.apply_chat_template_kwargs.enable_thinking=True' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
    actor_rollout_ref.model.enable_activation_offload=$ENABLE_ACTIVATION_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.clip_ratio=$EFFECTIVE_CLIP_RATIO \
    actor_rollout_ref.actor.clip_ratio_low=$EFFECTIVE_CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$EFFECTIVE_CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.clip_ratio_c=$EFFECTIVE_CLIP_RATIO_C \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$ROLLOUT_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.max_tokens=$ROLLOUT_MAX_TOKENS \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=false \
    trainer.default_local_dir="$OUTPUT_DIR" \
    +trainer.metrics_csv_freq=$METRICS_CSV_FREQ \
    +trainer.metrics_csv_filename="$METRICS_CSV_FILENAME" \
    trainer.project_name=textcraft_grpo_validated \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.resume_mode=disable \
    algorithm.optimize_prefix_tokens=$OPTIMIZE_PREFIX_TOKENS \
    algorithm.prefix_loss_weight=$PREFIX_LOSS_WEIGHT \
    algorithm.prefix_loss_mode=$PREFIX_LOSS_MODE \
    algorithm.prefix_advantage_mode=$PREFIX_ADVANTAGE_MODE \
    algorithm.prefix_advantage_constant=$PREFIX_ADVANTAGE_CONSTANT \
    +algorithm.prefix_cont_adv_weight=$PREFIX_CONT_ADV_WEIGHT \
    +algorithm.prefix_family_lift_weight=$PREFIX_FAMILY_LIFT_WEIGHT \
    +algorithm.prefix_family_lift_clip=$PREFIX_FAMILY_LIFT_CLIP \
    +algorithm.use_textcraft_bc_aux=$USE_TEXTCRAFT_BC_AUX \
    +algorithm.textcraft_bc_weight=$TEXTCRAFT_BC_WEIGHT \
    +algorithm.textcraft_bc_source=$TEXTCRAFT_BC_SOURCE \
    +algorithm.textcraft_bc_max_length=$TEXTCRAFT_BC_MAX_LENGTH_OVERRIDE \
    +algorithm.use_textcraft_teacher_demo=$USE_TEXTCRAFT_TEACHER_DEMO \
    +algorithm.textcraft_teacher_demo_weight=$TEXTCRAFT_TEACHER_DEMO_WEIGHT \
    +algorithm.textcraft_teacher_demo_labels="$HYDRA_TEXTCRAFT_TEACHER_DEMO_LABELS" \
    +algorithm.textcraft_teacher_demo_repeat_to_rollout_n=$TEXTCRAFT_TEACHER_DEMO_REPEAT_TO_ROLLOUT_N \
    +algorithm.textcraft_teacher_demo_skip_overlong=$TEXTCRAFT_TEACHER_DEMO_SKIP_OVERLONG \
    data.shuffle=$DATA_SHUFFLE \
    actor_rollout_ref.actor.prefix_clip_ratio=$EFFECTIVE_PREFIX_CLIP_RATIO \
    actor_rollout_ref.actor.prefix_clip_ratio_low=$EFFECTIVE_PREFIX_CLIP_RATIO_LOW \
    actor_rollout_ref.actor.prefix_clip_ratio_high=$EFFECTIVE_PREFIX_CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.prefix_clip_ratio_c=$EFFECTIVE_PREFIX_CLIP_RATIO_C \
    2>&1 | tee -a "$LOG_FILE"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "错误: 训练失败，退出码: $TRAIN_EXIT_CODE" | tee -a "$LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "训练完成！" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "检查点目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

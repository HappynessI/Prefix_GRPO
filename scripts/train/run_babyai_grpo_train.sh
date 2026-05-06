#!/bin/bash
set -eo pipefail

# BabyAI GRPO 训练脚本
# 环境服务与训练进程在同一个 Pod 内启动，训练脚本自动管理环境服务生命周期。

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT=${VERL_ROOT:-"${PROJECT_ROOT}/verl"}
CONFIG_ROOT=${CONFIG_ROOT:-"${PROJECT_ROOT}/config"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/Qwen3-1.7B"}
DATA_PATH_EXPLICIT=${DATA_PATH+x}
VAL_DATA_PATH_EXPLICIT=${VAL_DATA_PATH+x}
OUTPUT_DIR_EXPLICIT=${OUTPUT_DIR+x}
PREFIX_REPLAY_ONLY=${PREFIX_REPLAY_ONLY:-false}

case "${PREFIX_REPLAY_ONLY,,}" in
    true|1|yes|y)
        PREFIX_REPLAY_ONLY_ENABLED=true
        ;;
    false|0|no|n)
        PREFIX_REPLAY_ONLY_ENABLED=false
        ;;
    *)
        echo "错误: PREFIX_REPLAY_ONLY=$PREFIX_REPLAY_ONLY 非法，必须是 true/false。" >&2
        exit 1
        ;;
esac

if [ "$PREFIX_REPLAY_ONLY_ENABLED" = "true" ] && [ -z "$DATA_PATH_EXPLICIT" ]; then
    DATA_PATH="${DATA_ROOT}/babyai/prefix-rl/replay_validated/main_change_top3_w11_fullflow.parquet"
else
    DATA_PATH=${DATA_PATH:-"${DATA_ROOT}/babyai/train.parquet"}
fi
if [ "$PREFIX_REPLAY_ONLY_ENABLED" = "true" ] && [ -z "$VAL_DATA_PATH_EXPLICIT" ]; then
    VAL_DATA_PATH="$DATA_PATH"
else
    VAL_DATA_PATH=${VAL_DATA_PATH:-"${DATA_ROOT}/babyai/train.parquet"}
fi
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/babyai_grpo"}
INTERACTION_CONFIG=${INTERACTION_CONFIG:-"${CONFIG_ROOT}/interaction_config/babyai_interaction.yaml"}
AGENTGYM_ROOT=${AGENTGYM_ROOT:-"${PROJECT_ROOT}/envs/AgentGym"}
WHEEL_DIR=${WHEEL_DIR:-"${PROJECT_ROOT}/third_party/wheels_babyai"}
RUNTIME_REQS=${RUNTIME_REQS:-"${PROJECT_ROOT}/third_party/requirements_babyai_runtime.txt"}
VERL_WHEEL_DIR=${VERL_WHEEL_DIR:-"${PROJECT_ROOT}/third_party/wheels_verl_py312"}
VERL_RUNTIME_REQS=${VERL_RUNTIME_REQS:-"${PROJECT_ROOT}/third_party/requirements_verl_runtime.txt"}
BABYAI_PORT=${BABYAI_PORT:-36002}
BABYAI_SERVER_COUNT=${BABYAI_SERVER_COUNT:-1}
BABYAI_SERVER="http://127.0.0.1:${BABYAI_PORT}"
BABYAI_LOG=""

# NUM_GPUS 由 oss-submit.sh 通过 --env 传入，GPU_IDS 未传入时根据 NUM_GPUS 自动生成
NUM_GPUS=${NUM_GPUS:-2}
if [ -z "${GPU_IDS:-}" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

NUM_EPOCHS=${NUM_EPOCHS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
PPO_EPOCHS=${PPO_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
CLIP_RATIO=${CLIP_RATIO:-}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
USE_KL_LOSS=${USE_KL_LOSS:-false}
ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-false}
SAVE_FREQ=${SAVE_FREQ:-500}
TEST_FREQ=${TEST_FREQ:--1}
ROLLOUT_N=${ROLLOUT_N:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.80}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-4096}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-$MAX_RESPONSE_LENGTH}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-512}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-20}
MAX_USER_TURNS=${MAX_USER_TURNS:-21}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}
VAL_N=${VAL_N:-1}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
METRICS_CSV_FREQ=${METRICS_CSV_FREQ:-50}
METRICS_CSV_FILENAME=${METRICS_CSV_FILENAME:-training_metrics.csv}
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}

OPTIMIZE_PREFIX_TOKENS=${OPTIMIZE_PREFIX_TOKENS:-false}
if [ "$PREFIX_REPLAY_ONLY_ENABLED" = "true" ]; then
    case "${OPTIMIZE_PREFIX_TOKENS,,}" in
        false|0|no|n)
            OPTIMIZE_PREFIX_TOKENS=false
            ;;
        *)
            echo "错误: PREFIX_REPLAY_ONLY=true 表示只 replay prefix actions，不学习 prefix；OPTIMIZE_PREFIX_TOKENS 必须为 false。" >&2
            exit 1
            ;;
    esac
fi
PREFIX_LOSS_WEIGHT=${PREFIX_LOSS_WEIGHT:-1.0}
PREFIX_LOSS_MODE=${PREFIX_LOSS_MODE:-split}
PREFIX_ADVANTAGE_MODE=${PREFIX_ADVANTAGE_MODE:-cont_mean_abs}
PREFIX_ADVANTAGE_CONSTANT=${PREFIX_ADVANTAGE_CONSTANT:-1.0}
PREFIX_CONT_ADV_WEIGHT=${PREFIX_CONT_ADV_WEIGHT:-1.0}
PREFIX_FAMILY_LIFT_WEIGHT=${PREFIX_FAMILY_LIFT_WEIGHT:-1.0}
PREFIX_FAMILY_LIFT_CLIP=${PREFIX_FAMILY_LIFT_CLIP:-1.0}
USE_BC_AUX=${USE_BC_AUX:-false}
BC_WEIGHT=${BC_WEIGHT:-0.0}
BC_SOURCE=${BC_SOURCE:-prefix}
BC_MAX_LENGTH=${BC_MAX_LENGTH:-null}
USE_TEACHER_DEMO=${USE_TEACHER_DEMO:-false}
TEACHER_DEMO_WEIGHT=${TEACHER_DEMO_WEIGHT:-1.0}
TEACHER_DEMO_LABELS=${TEACHER_DEMO_LABELS:-teacher_demo,demo}
TEACHER_DEMO_REPEAT_TO_ROLLOUT_N=${TEACHER_DEMO_REPEAT_TO_ROLLOUT_N:-true}
TEACHER_DEMO_SKIP_OVERLONG=${TEACHER_DEMO_SKIP_OVERLONG:-false}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
GRPO_MIS_ENABLE=${GRPO_MIS_ENABLE:-false}
ROLLOUT_IS=${ROLLOUT_IS:-"sequence"}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
ROLLOUT_RS=${ROLLOUT_RS:-"sequence"}
ROLLOUT_RS_THRESHOLD=${ROLLOUT_RS_THRESHOLD:-2.0}
ROLLOUT_RS_THRESHOLD_LOWER=${ROLLOUT_RS_THRESHOLD_LOWER:-0.2}

case "${GRPO_MIS_ENABLE,,}" in
    true|1|yes|y)
        GRPO_MIS_ENABLED=true
        ;;
    false|0|no|n)
        GRPO_MIS_ENABLED=false
        ;;
    *)
        echo "错误: GRPO_MIS_ENABLE=$GRPO_MIS_ENABLE 非法，必须是 true/false。" >&2
        exit 1
        ;;
esac
if [ -z "$OUTPUT_DIR_EXPLICIT" ]; then
    if [ "$GRPO_MIS_ENABLED" = "true" ] && [ "$PREFIX_REPLAY_ONLY_ENABLED" = "true" ]; then
        OUTPUT_DIR="${OUTPUT_ROOT}/babyai_grpo_mis_prefix_replay_only"
    elif [ "$GRPO_MIS_ENABLED" = "true" ]; then
        OUTPUT_DIR="${OUTPUT_ROOT}/babyai_grpo_mis"
    elif [ "$PREFIX_REPLAY_ONLY_ENABLED" = "true" ]; then
        OUTPUT_DIR="${OUTPUT_ROOT}/babyai_grpo_prefix_replay_only"
    fi
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"
BABYAI_LOG="$LOG_DIR/babyai_env_${TIMESTAMP}.log"

GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    echo "错误: GPU_IDS中的GPU数量($GPU_COUNT)与NUM_GPUS($NUM_GPUS)不一致！" | tee -a "$LOG_FILE"
    exit 1
fi
if ! [[ "$BABYAI_SERVER_COUNT" =~ ^[0-9]+$ ]] || [ "$BABYAI_SERVER_COUNT" -lt 1 ]; then
    echo "错误: BABYAI_SERVER_COUNT=$BABYAI_SERVER_COUNT 非法，必须是 >= 1 的整数。" | tee -a "$LOG_FILE"
    exit 1
fi
case "$RESUME_MODE" in
    auto|disable|resume_path) ;;
    *)
        echo "错误: RESUME_MODE=$RESUME_MODE 非法，必须是 auto、disable 或 resume_path。" | tee -a "$LOG_FILE"
        exit 1
        ;;
esac
if [ "$RESUME_MODE" = "resume_path" ]; then
    if [ -z "$RESUME_FROM_PATH" ] || [ "$RESUME_FROM_PATH" = "null" ]; then
        echo "错误: RESUME_MODE=resume_path 时必须设置 RESUME_FROM_PATH。" | tee -a "$LOG_FILE"
        exit 1
    fi
    RESUME_FROM_PATH="${RESUME_FROM_PATH%/}"
    if [[ "$RESUME_FROM_PATH" == oss://* ]]; then
        echo "错误: verl PPO resume 需要本地/PVC路径，不能直接使用 oss:// 路径。" | tee -a "$LOG_FILE"
        echo "请改用 /oss-pvc/zhk_wyh/.../global_step_xxx 形式。" | tee -a "$LOG_FILE"
        exit 1
    fi
    if [[ "$RESUME_FROM_PATH" != *global_step_* ]]; then
        echo "错误: RESUME_FROM_PATH 必须指向包含 global_step_ 的 checkpoint 目录。" | tee -a "$LOG_FILE"
        exit 1
    fi
    if [ ! -d "$RESUME_FROM_PATH/actor" ]; then
        echo "错误: checkpoint actor 目录不存在: $RESUME_FROM_PATH/actor" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

cleanup_babyai() {
    for pid in "${BABYAI_PIDS[@]:-}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "" | tee -a "$LOG_FILE"
            echo "清理 BabyAI 环境服务 (PID=$pid)..." | tee -a "$LOG_FILE"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "BabyAI 环境服务已清理。" | tee -a "$LOG_FILE"
}
trap cleanup_babyai EXIT

echo "============================================" | tee -a "$LOG_FILE"
echo "正在当前 Pod 内启动 BabyAI 环境服务" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

echo "离线安装 BabyAI 运行时依赖..." | tee -a "$LOG_FILE"
echo "  wheel 目录: $WHEEL_DIR" | tee -a "$LOG_FILE"
echo "  requirements: $RUNTIME_REQS" | tee -a "$LOG_FILE"

if [ ! -d "$WHEEL_DIR" ]; then
    echo "错误: wheel 目录不存在: $WHEEL_DIR" | tee -a "$LOG_FILE"
    echo "请在开发机运行 scripts/prepare_babyai_wheels.sh 并重新上传代码。" | tee -a "$LOG_FILE"
    exit 1
fi

WHEEL_COUNT=$(find "$WHEEL_DIR" -maxdepth 1 -name "*.whl" -type f 2>/dev/null | wc -l)
if [ "$WHEEL_COUNT" -eq 0 ]; then
    echo "错误: wheel 目录为空: $WHEEL_DIR" | tee -a "$LOG_FILE"
    echo "请在开发机运行 scripts/prepare_babyai_wheels.sh 并重新上传代码。" | tee -a "$LOG_FILE"
    exit 1
fi

echo "  发现 $WHEEL_COUNT 个 wheel 文件" | tee -a "$LOG_FILE"

echo "离线安装 pip/setuptools/wheel（支持本地 editable 安装）..." | tee -a "$LOG_FILE"
if ! python3 -m pip install \
        --no-index \
        --find-links="$WHEEL_DIR" \
        pip setuptools wheel \
        2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: pip/setuptools/wheel 离线安装失败。" | tee -a "$LOG_FILE"
    echo "wheel 目录: $WHEEL_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

if ! python3 -m pip install \
        --no-index \
        --find-links="$WHEEL_DIR" \
        -r "$RUNTIME_REQS" \
        2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: BabyAI 运行时依赖离线安装失败。" | tee -a "$LOG_FILE"
    echo "提示: 请优先确认以下两点：" | tee -a "$LOG_FILE"
    echo "  1. 在开发机已执行 scripts/prepare_babyai_wheels.sh 并成功下载完整依赖树" | tee -a "$LOG_FILE"
    echo "  2. third_party/wheels_babyai/ 中确实存在 .whl 文件后再重新上传 OSS" | tee -a "$LOG_FILE"
    echo "wheel 目录: $WHEEL_DIR" | tee -a "$LOG_FILE"
    echo "requirements: $RUNTIME_REQS" | tee -a "$LOG_FILE"
    exit 1
fi
echo "BabyAI 运行时依赖安装完成" | tee -a "$LOG_FILE"

echo "安装 agentenv-babyai..." | tee -a "$LOG_FILE"
cd "$AGENTGYM_ROOT/agentenv-babyai"
if ! python3 -m pip install --no-build-isolation -e . --no-deps -q 2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: agentenv-babyai 安装失败" | tee -a "$LOG_FILE"
    exit 1
fi
echo "agentenv-babyai 安装完成" | tee -a "$LOG_FILE"

echo "启动 BabyAI 服务池 (base_port=${BABYAI_PORT}, count=${BABYAI_SERVER_COUNT})..." | tee -a "$LOG_FILE"
BABYAI_PIDS=()
BABYAI_SERVERS=()
BABYAI_LOGS=()
for server_idx in $(seq 0 $((BABYAI_SERVER_COUNT - 1))); do
    port=$((BABYAI_PORT + server_idx))
    server_url="http://127.0.0.1:${port}"
    server_log="$LOG_DIR/babyai_env_${port}_${TIMESTAMP}.log"
    BABYAI_SERVERS+=("$server_url")
    BABYAI_LOGS+=("$server_log")
    babyai --host 0.0.0.0 --port ${port} > "$server_log" 2>&1 &
    BABYAI_PIDS+=("$!")
    echo "BabyAI server[$server_idx] PID=${BABYAI_PIDS[$server_idx]}, url=$server_url, log=$server_log" | tee -a "$LOG_FILE"
done
BABYAI_SERVER="${BABYAI_SERVERS[0]}"
BABYAI_SERVERS_CSV="$(IFS=,; echo "${BABYAI_SERVERS[*]}")"
BABYAI_LOG="${BABYAI_LOGS[0]}"

# 等待服务池 ready（最多 120 秒）
for server_idx in "${!BABYAI_SERVERS[@]}"; do
    server_url="${BABYAI_SERVERS[$server_idx]}"
    server_log="${BABYAI_LOGS[$server_idx]}"
    READY=false
    for i in $(seq 1 60); do
        if curl -sf "${server_url}/" > /dev/null 2>&1; then
            READY=true
            break
        fi
        echo "  等待 BabyAI server[$server_idx] 启动... ($i/60)" | tee -a "$LOG_FILE"
        sleep 2
    done

    if [ "$READY" != "true" ]; then
        echo "错误: BabyAI server[$server_idx] 启动超时（120秒），请查看 $server_log" | tee -a "$LOG_FILE"
        exit 1
    fi
done
echo "BabyAI 服务池已就绪: ${BABYAI_SERVERS_CSV}" | tee -a "$LOG_FILE"

RUNTIME_INTERACTION_CONFIG="$LOG_DIR/babyai_interaction_${BABYAI_PORT}_${TIMESTAMP}.yaml"
cat > "$RUNTIME_INTERACTION_CONFIG" <<EOF
interaction:
  - name: babyai
    class_name: verl.interactions.babyai_interaction.BabyAIInteraction
    config:
      env_server_base: "$BABYAI_SERVER"
      env_server_bases:
$(for server_url in "${BABYAI_SERVERS[@]}"; do printf '        - "%s"\n' "$server_url"; done)
      timeout: 600
      max_retries: 3
EOF
INTERACTION_CONFIG="$RUNTIME_INTERACTION_CONFIG"
echo "运行时 BabyAI interaction config: $INTERACTION_CONFIG" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "安装 verl 框架..." | tee -a "$LOG_FILE"
cd "$VERL_ROOT"
echo "离线安装 verl 运行时依赖..." | tee -a "$LOG_FILE"
echo "  verl wheel 目录: $VERL_WHEEL_DIR" | tee -a "$LOG_FILE"
echo "  verl requirements: $VERL_RUNTIME_REQS" | tee -a "$LOG_FILE"
if [ ! -d "$VERL_WHEEL_DIR" ]; then
    echo "错误: verl wheel 目录不存在: $VERL_WHEEL_DIR" | tee -a "$LOG_FILE"
    echo "请在开发机运行 scripts/prepare_verl_wheels_py312.sh 并重新上传代码。" | tee -a "$LOG_FILE"
    exit 1
fi
VERL_WHEEL_COUNT=$(find "$VERL_WHEEL_DIR" -maxdepth 1 -name "*.whl" -type f 2>/dev/null | wc -l)
if [ "$VERL_WHEEL_COUNT" -eq 0 ]; then
    echo "错误: verl wheel 目录为空: $VERL_WHEEL_DIR" | tee -a "$LOG_FILE"
    echo "请在开发机运行 scripts/prepare_verl_wheels_py312.sh 并重新上传代码。" | tee -a "$LOG_FILE"
    exit 1
fi
if ! python3 -m pip install \
        --no-index \
        --find-links="$VERL_WHEEL_DIR" \
        -r "$VERL_RUNTIME_REQS" \
        2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: verl 运行时依赖离线安装失败。" | tee -a "$LOG_FILE"
    echo "verl wheel 目录: $VERL_WHEEL_DIR" | tee -a "$LOG_FILE"
    echo "verl requirements: $VERL_RUNTIME_REQS" | tee -a "$LOG_FILE"
    exit 1
fi
if ! python3 -m pip install --no-build-isolation -e . --no-deps -q 2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: verl 安装失败" | tee -a "$LOG_FILE"
    exit 1
fi
echo "verl 安装完成" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "获取 verl 内部配置路径..." | tee -a "$LOG_FILE"
VERL_CONFIG_ROOT="${VERL_ROOT}/verl/trainer/config"
if [ ! -d "$VERL_CONFIG_ROOT" ]; then
    echo "错误: verl 配置目录不存在: $VERL_CONFIG_ROOT" | tee -a "$LOG_FILE"
    exit 1
fi
echo "verl 配置路径: $VERL_CONFIG_ROOT" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "校验 BabyAI 训练数据协议..." | tee -a "$LOG_FILE"
PREFIX_REPLAY_FILTERED_PATH="$OUTPUT_DIR/prefix_replay_only_data/train_prefix_replay_only.parquet"
export DATA_PATH
export PREFIX_REPLAY_ONLY_ENABLED
export PREFIX_REPLAY_FILTERED_PATH
python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

data_path = os.environ["DATA_PATH"]
prefix_replay_only = os.environ["PREFIX_REPLAY_ONLY_ENABLED"].lower() == "true"
filtered_path = Path(os.environ["PREFIX_REPLAY_FILTERED_PATH"])

if not os.path.exists(data_path):
    raise FileNotFoundError(f"训练数据不存在: {data_path}")

df = pd.read_parquet(data_path)
if len(df) == 0:
    raise ValueError(f"训练数据为空: {data_path}")

required_columns = ["data_source", "prompt", "reward_model", "extra_info"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"缺失 GRPO 必需列: {missing}. data_path={data_path}")


def optional_int(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def list_len(value):
    if value is None or isinstance(value, (str, bytes)):
        return 0
    if isinstance(value, Sequence):
        return len(value)
    if hasattr(value, "tolist"):
        return len(value.tolist())
    return 0


bad_rows = []
drop_rows = []
prefix_action_len_counter = Counter()
strategy_counter = Counter()
replay_counter = Counter()
prefix_rows = 0
raw_rows = 0

for idx, row in df.iterrows():
    row_dict = row.to_dict()
    extra_info = row_dict.get("extra_info", {})
    if not isinstance(extra_info, dict):
        bad_rows.append((idx, "extra_info_not_dict"))
        continue
    interaction_kwargs = extra_info.get("interaction_kwargs", {})
    if not isinstance(interaction_kwargs, dict):
        bad_rows.append((idx, "interaction_kwargs_not_dict"))
        continue

    strategy = row_dict.get("strategy", extra_info.get("strategy", "unknown"))
    replay_category = row_dict.get("replay_category", extra_info.get("replay_category", "unknown"))
    strategy_counter[str(strategy)] += 1
    replay_counter[str(replay_category)] += 1

    prefix_token_count = optional_int(row_dict.get("prefix_token_count")) if "prefix_token_count" in row_dict else None
    prefix_actions_len = list_len(interaction_kwargs.get("prefix_actions", None))
    is_raw_row = (
        prefix_token_count == 0
        or (prefix_token_count is None and str(strategy).lower() == "raw")
        or (prefix_token_count is None and str(replay_category).lower() == "raw")
    )

    if is_raw_row:
        raw_rows += 1
    else:
        prefix_rows += 1

    if prefix_actions_len > 0:
        prefix_action_len_counter[prefix_actions_len] += 1

    if prefix_replay_only:
        if is_raw_row and prefix_actions_len != 0:
            bad_rows.append((idx, "raw_row_prefix_actions_not_empty"))
        if not is_raw_row and prefix_actions_len == 0:
            drop_rows.append(idx)

if bad_rows:
    raise ValueError(f"BabyAI prefix replay-only 数据协议非法，前5条={bad_rows[:5]}")
if prefix_replay_only and drop_rows:
    filtered_df = df.drop(index=drop_rows).reset_index(drop=True)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(filtered_path)
    prefix_rows -= len(drop_rows)
    print(f"  dropped_prefix_rows_without_actions: {len(drop_rows)}")
    print(f"  filtered_data_path: {filtered_path}")
if prefix_replay_only and prefix_rows == 0:
    raise ValueError("PREFIX_REPLAY_ONLY=true 但数据中没有 prefix rows。")
if prefix_replay_only and not prefix_action_len_counter:
    raise ValueError("PREFIX_REPLAY_ONLY=true 但没有任何非空 prefix_actions。")

mode = "prefix replay only" if prefix_replay_only else "standard grpo"
print(f"  模式: {mode}")
print(f"  样本数: {len(df)}")
print(f"  raw_rows: {raw_rows}, prefix_rows: {prefix_rows}")
print(f"  strategy分布: {dict(strategy_counter)}")
print(f"  replay_category分布: {dict(replay_counter)}")
print(f"  prefix_actions长度分布: {dict(prefix_action_len_counter)}")
print("  数据校验通过")
PY
DATA_CHECK_EXIT_CODE=${PIPESTATUS[0]}
if [ "$DATA_CHECK_EXIT_CODE" -ne 0 ]; then
    echo "错误: BabyAI 训练数据协议校验失败。" | tee -a "$LOG_FILE"
    exit $DATA_CHECK_EXIT_CODE
fi
if [ "$PREFIX_REPLAY_ONLY_ENABLED" = "true" ] && [ -f "$PREFIX_REPLAY_FILTERED_PATH" ]; then
    DATA_PATH="$PREFIX_REPLAY_FILTERED_PATH"
    if [ -z "$VAL_DATA_PATH_EXPLICIT" ]; then
        VAL_DATA_PATH="$DATA_PATH"
    fi
    echo "BabyAI replay-only 训练数据已过滤: $DATA_PATH" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "BabyAI GRPO 训练 - Qwen3-1.7B" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "训练数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "Interaction Config: $INTERACTION_CONFIG" | tee -a "$LOG_FILE"
echo "BabyAI servers: $BABYAI_SERVERS_CSV" | tee -a "$LOG_FILE"
echo "输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Prefix replay only: $PREFIX_REPLAY_ONLY_ENABLED" | tee -a "$LOG_FILE"
echo "GPU IDs: $GPU_IDS, 数量: $NUM_GPUS" | tee -a "$LOG_FILE"
echo "Epochs: $NUM_EPOCHS, Batch: $TRAIN_BATCH_SIZE, LR: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "PPO: epochs=$PPO_EPOCHS, mini_batch=$PPO_MINI_BATCH_SIZE, micro_batch=$MICRO_BATCH_SIZE, rollout_n=$ROLLOUT_N" | tee -a "$LOG_FILE"
echo "Current training defaults: prompt=$MAX_PROMPT_LENGTH, rollout_prompt=$ROLLOUT_PROMPT_LENGTH, cumulative_response=$MAX_RESPONSE_LENGTH, rollout_response=$ROLLOUT_RESPONSE_LENGTH, per_turn_max=$ROLLOUT_MAX_TOKENS, max_model_len=$MAX_MODEL_LEN, ppo_max_token_len=$PPO_MAX_TOKEN_LEN, max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS, turns=${MAX_USER_TURNS}/${MAX_ASSISTANT_TURNS}, val_sample=${VAL_DO_SAMPLE}/${VAL_N}" | tee -a "$LOG_FILE"
echo "Regularization: use_kl_loss=$USE_KL_LOSS, entropy_coeff=$ENTROPY_COEFF" | tee -a "$LOG_FILE"
echo "Memory: enable_activation_offload=$ENABLE_ACTIVATION_OFFLOAD" | tee -a "$LOG_FILE"
echo "Prefix/BC/demo switches: optimize_prefix_tokens=$OPTIMIZE_PREFIX_TOKENS, prefix_loss_mode=$PREFIX_LOSS_MODE, prefix_advantage_mode=$PREFIX_ADVANTAGE_MODE, use_bc_aux=$USE_BC_AUX, use_teacher_demo=$USE_TEACHER_DEMO" | tee -a "$LOG_FILE"
echo "GRPO-MIS: enabled=$GRPO_MIS_ENABLED, IS=$ROLLOUT_IS, IS_threshold=$ROLLOUT_IS_THRESHOLD, RS=$ROLLOUT_RS, RS_threshold=$ROLLOUT_RS_THRESHOLD, RS_threshold_lower=$ROLLOUT_RS_THRESHOLD_LOWER" | tee -a "$LOG_FILE"
echo "Metrics CSV: $OUTPUT_DIR/metrics/$METRICS_CSV_FILENAME (every $METRICS_CSV_FREQ steps)" | tee -a "$LOG_FILE"
echo "Resume: mode=$RESUME_MODE, from=$RESUME_FROM_PATH" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

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

PROJECT_NAME="babyai_grpo"
if [ "$GRPO_MIS_ENABLED" = "true" ]; then
    PROJECT_NAME="babyai_grpo_mis"
fi
if [ "$PREFIX_REPLAY_ONLY_ENABLED" = "true" ]; then
    PROJECT_NAME="${PROJECT_NAME}_prefix_replay_only"
fi
EXPERIMENT_NAME="${PROJECT_NAME}_${TIMESTAMP}"

ACTOR_CLIP_OVERRIDE_ARGS=()
if [ -n "${CLIP_RATIO}" ]; then
    ACTOR_CLIP_OVERRIDE_ARGS+=("actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}")
fi
if [ -n "${CLIP_RATIO_LOW}" ]; then
    ACTOR_CLIP_OVERRIDE_ARGS+=("actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}")
fi
if [ -n "${CLIP_RATIO_HIGH}" ]; then
    ACTOR_CLIP_OVERRIDE_ARGS+=("actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}")
fi
HYDRA_TEACHER_DEMO_LABELS="'${TEACHER_DEMO_LABELS}'"

GRPO_MIS_OVERRIDE_ARGS=()
if [ "$GRPO_MIS_ENABLED" = "true" ]; then
    GRPO_MIS_OVERRIDE_ARGS+=(
        "actor_rollout_ref.actor.calculate_entropy=true"
        "actor_rollout_ref.rollout.calculate_log_probs=true"
        "algorithm.rollout_correction.bypass_mode=true"
        "algorithm.rollout_correction.use_policy_gradient=true"
        "algorithm.rollout_correction.rollout_is=${ROLLOUT_IS}"
        "algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD}"
        "algorithm.rollout_correction.rollout_rs=${ROLLOUT_RS}"
        "algorithm.rollout_correction.rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD}"
        "algorithm.rollout_correction.rollout_rs_threshold_lower=${ROLLOUT_RS_THRESHOLD_LOWER}"
    )
fi

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_ROOT}" \
    --config-name='babyai_grpo_train' \
    hydra.searchpath=[file://${VERL_CONFIG_ROOT},file://${CONFIG_ROOT}] \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    '+data.apply_chat_template_kwargs.enable_thinking=True' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=$ENABLE_ACTIVATION_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    "${ACTOR_CLIP_OVERRIDE_ARGS[@]}" \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    "${GRPO_MIS_OVERRIDE_ARGS[@]}" \
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
    algorithm.optimize_prefix_tokens=$OPTIMIZE_PREFIX_TOKENS \
    algorithm.prefix_loss_weight=$PREFIX_LOSS_WEIGHT \
    algorithm.prefix_loss_mode=$PREFIX_LOSS_MODE \
    algorithm.prefix_advantage_mode=$PREFIX_ADVANTAGE_MODE \
    algorithm.prefix_advantage_constant=$PREFIX_ADVANTAGE_CONSTANT \
    algorithm.prefix_cont_adv_weight=$PREFIX_CONT_ADV_WEIGHT \
    algorithm.prefix_family_lift_weight=$PREFIX_FAMILY_LIFT_WEIGHT \
    algorithm.prefix_family_lift_clip=$PREFIX_FAMILY_LIFT_CLIP \
    algorithm.use_bc_aux=$USE_BC_AUX \
    algorithm.bc_weight=$BC_WEIGHT \
    algorithm.bc_source=$BC_SOURCE \
    algorithm.bc_max_length=$BC_MAX_LENGTH \
    algorithm.use_teacher_demo=$USE_TEACHER_DEMO \
    algorithm.teacher_demo_weight=$TEACHER_DEMO_WEIGHT \
    algorithm.teacher_demo_labels="$HYDRA_TEACHER_DEMO_LABELS" \
    algorithm.teacher_demo_repeat_to_rollout_n=$TEACHER_DEMO_REPEAT_TO_ROLLOUT_N \
    algorithm.teacher_demo_skip_overlong=$TEACHER_DEMO_SKIP_OVERLONG \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.resume_from_path="$RESUME_FROM_PATH" \
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

#!/bin/bash
set -eo pipefail

# TextCraft GRPO + TIS 训练脚本
# 环境服务与训练进程在同一个 Pod 内启动，训练脚本自动管理环境服务生命周期。

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/Qwen3-1.7B"}
DATA_PATH=${DATA_PATH:-"${DATA_ROOT}/textcraft/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/textcraft_grpo_tis"}
INTERACTION_CONFIG=${INTERACTION_CONFIG:-"${CONFIG_ROOT}/interaction_config/textcraft_interaction.yaml"}
VERL_ROOT=${VERL_ROOT:-"${PROJECT_ROOT}/verl"}
CONFIG_ROOT=${CONFIG_ROOT:-"${PROJECT_ROOT}/config"}
AGENTGYM_ROOT=${AGENTGYM_ROOT:-"${PROJECT_ROOT}/envs/AgentGym"}
WHEEL_DIR=${WHEEL_DIR:-"${PROJECT_ROOT}/third_party/wheels_py312"}
RUNTIME_REQS=${RUNTIME_REQS:-"${PROJECT_ROOT}/third_party/requirements_textcraft_runtime.txt"}
VERL_WHEEL_DIR=${VERL_WHEEL_DIR:-"${PROJECT_ROOT}/third_party/wheels_verl_py312"}
VERL_RUNTIME_REQS=${VERL_RUNTIME_REQS:-"${PROJECT_ROOT}/third_party/requirements_verl_runtime.txt"}
TEXTCRAFT_PORT=${TEXTCRAFT_PORT:-36001}
TEXTCRAFT_SERVER="http://127.0.0.1:${TEXTCRAFT_PORT}"
TEXTCRAFT_LOG=""

# NUM_GPUS 由 oss-submit.sh 通过 --env 传入，GPU_IDS 未传入时根据 NUM_GPUS 自动生成
NUM_GPUS=${NUM_GPUS:-2}
if [ -z "${GPU_IDS:-}" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

NUM_EPOCHS=${NUM_EPOCHS:-30}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_EPOCHS=${PPO_EPOCHS:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
SAVE_FREQ=${SAVE_FREQ:-200}
TEST_FREQ=${TEST_FREQ:--1}
ROLLOUT_N=${ROLLOUT_N:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-10240}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-10240}
ROLLOUT_MAX_TOKENS=${ROLLOUT_MAX_TOKENS:-512}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-16384}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-30}
MAX_USER_TURNS=${MAX_USER_TURNS:-30}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}
VAL_N=${VAL_N:-1}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
METRICS_CSV_FREQ=${METRICS_CSV_FREQ:-50}
METRICS_CSV_FILENAME=${METRICS_CSV_FILENAME:-training_metrics.csv}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
ROLLOUT_IS=${ROLLOUT_IS:-"sequence"}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"
TEXTCRAFT_LOG="$LOG_DIR/textcraft_env_${TIMESTAMP}.log"

GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    echo "错误: GPU_IDS中的GPU数量($GPU_COUNT)与NUM_GPUS($NUM_GPUS)不一致！" | tee -a "$LOG_FILE"
    exit 1
fi

cleanup_textcraft() {
    if [ -n "$TEXTCRAFT_PID" ] && kill -0 "$TEXTCRAFT_PID" 2>/dev/null; then
        echo "" | tee -a "$LOG_FILE"
        echo "清理 TextCraft 环境服务 (PID=$TEXTCRAFT_PID)..." | tee -a "$LOG_FILE"
        kill "$TEXTCRAFT_PID" 2>/dev/null || true
        wait "$TEXTCRAFT_PID" 2>/dev/null || true
        echo "TextCraft 环境服务已清理。" | tee -a "$LOG_FILE"
    fi
}
trap cleanup_textcraft EXIT

echo "============================================" | tee -a "$LOG_FILE"
echo "正在当前 Pod 内启动 TextCraft 环境服务" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

echo "正在当前 Pod 内启动 TextCraft 环境服务" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

echo "离线安装 TextCraft 运行时依赖..." | tee -a "$LOG_FILE"
echo "  wheel 目录: $WHEEL_DIR" | tee -a "$LOG_FILE"
echo "  requirements: $RUNTIME_REQS" | tee -a "$LOG_FILE"

if [ ! -d "$WHEEL_DIR" ]; then
    echo "错误: wheel 目录不存在: $WHEEL_DIR" | tee -a "$LOG_FILE"
    echo "请在开发机运行 scripts/prepare_textcraft_wheels.sh 并重新上传代码。" | tee -a "$LOG_FILE"
    exit 1
fi

WHEEL_COUNT=$(find "$WHEEL_DIR" -maxdepth 1 -name "*.whl" -type f 2>/dev/null | wc -l)
if [ "$WHEEL_COUNT" -eq 0 ]; then
    echo "错误: wheel 目录为空: $WHEEL_DIR" | tee -a "$LOG_FILE"
    echo "请在开发机运行 scripts/prepare_textcraft_wheels.sh 并重新上传代码。" | tee -a "$LOG_FILE"
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
    echo "错误: TextCraft 运行时依赖离线安装失败。" | tee -a "$LOG_FILE"
    echo "提示: 请优先确认以下两点：" | tee -a "$LOG_FILE"
    echo "  1. 在开发机已执行 scripts/prepare_textcraft_wheels.sh 并成功下载完整依赖树" | tee -a "$LOG_FILE"
    echo "  2. third_party/wheels/ 中确实存在 .whl 文件后再重新上传 OSS" | tee -a "$LOG_FILE"
    echo "wheel 目录: $WHEEL_DIR" | tee -a "$LOG_FILE"
    echo "requirements: $RUNTIME_REQS" | tee -a "$LOG_FILE"
    exit 1
fi
echo "TextCraft 运行时依赖安装完成" | tee -a "$LOG_FILE"

echo "安装 agentenv-textcraft..." | tee -a "$LOG_FILE"
cd "$AGENTGYM_ROOT/agentenv-textcraft"
if ! python3 -m pip install --no-build-isolation -e . --no-deps -q 2>&1 | tee -a "$LOG_FILE"; then
    echo "错误: agentenv-textcraft 安装失败" | tee -a "$LOG_FILE"
    exit 1
fi
echo "agentenv-textcraft 安装完成" | tee -a "$LOG_FILE"

echo "启动 TextCraft 服务 (${TEXTCRAFT_SERVER})..." | tee -a "$LOG_FILE"
textcraft --host 0.0.0.0 --port ${TEXTCRAFT_PORT} > "$TEXTCRAFT_LOG" 2>&1 &
TEXTCRAFT_PID=$!
echo "TextCraft 进程 PID=$TEXTCRAFT_PID，日志: $TEXTCRAFT_LOG" | tee -a "$LOG_FILE"

# 等待服务 ready（最多 120 秒）
READY=false
for i in $(seq 1 60); do
    if curl -sf "${TEXTCRAFT_SERVER}/" > /dev/null 2>&1; then
        READY=true
        break
    fi
    echo "  等待 TextCraft 服务启动... ($i/60)" | tee -a "$LOG_FILE"
    sleep 2
done

if [ "$READY" != "true" ]; then
    echo "错误: TextCraft 环境服务启动超时（120秒），请查看 $TEXTCRAFT_LOG" | tee -a "$LOG_FILE"
    exit 1
fi
echo "TextCraft 环境服务已就绪 (${TEXTCRAFT_SERVER})" | tee -a "$LOG_FILE"
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
echo "============================================" | tee -a "$LOG_FILE"
echo "TextCraft GRPO+TIS 训练" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "训练数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "Interaction Config: $INTERACTION_CONFIG" | tee -a "$LOG_FILE"
echo "输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "GPU IDs: $GPU_IDS, 数量: $NUM_GPUS" | tee -a "$LOG_FILE"
echo "Epochs: $NUM_EPOCHS, Batch: $TRAIN_BATCH_SIZE, LR: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "Metrics CSV: $OUTPUT_DIR/metrics/$METRICS_CSV_FILENAME (every $METRICS_CSV_FREQ steps)" | tee -a "$LOG_FILE"
echo "TIS: IS=$ROLLOUT_IS" | tee -a "$LOG_FILE"
echo "Current training defaults: prompt=$MAX_PROMPT_LENGTH, cumulative_response=$MAX_RESPONSE_LENGTH, per_turn_max=$ROLLOUT_MAX_TOKENS, max_model_len=$MAX_MODEL_LEN" | tee -a "$LOG_FILE"
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

EXPERIMENT_NAME="textcraft_grpo_tis_${TIMESTAMP}"

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
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.calculate_entropy=true \
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
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    algorithm.rollout_correction.bypass_mode=true \
    algorithm.rollout_correction.use_policy_gradient=true \
    algorithm.rollout_correction.rollout_is=$ROLLOUT_IS \
    algorithm.rollout_correction.rollout_is_threshold=$ROLLOUT_IS_THRESHOLD \
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
    trainer.project_name=textcraft_grpo_tis \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.resume_mode=disable \
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

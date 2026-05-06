#!/bin/bash
set -euo pipefail

if [ "${SFT_DEBUG:-false}" = "true" ]; then
    set -x
fi

# ============================================================================
# AgentGym SFT training launcher for H200 codebase.
#
# Defaults intentionally follow env-specific main datasets.  SciWorld uses the
# max4096 filtered success subset to drop the single 11898-token outlier;
# BabyAI/ALFWorld still default to the full success_only datasets.  The synced
# H200 upload copy reads datasets from ${PROJECT_ROOT}/datasets/SFT-Data by
# default; override SFT_DATA_ROOT if needed.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_ROOT="${PROJECT_ROOT}/verl"
SFT_DATA_ROOT=${SFT_DATA_ROOT:-${PROJECT_ROOT}/datasets/SFT-Data}
SFT_OUTPUT_ROOT=${SFT_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}
VERL_WHEEL_DIR=${VERL_WHEEL_DIR:-${PROJECT_ROOT}/third_party/wheels_verl_py312}
VERL_RUNTIME_REQS=${VERL_RUNTIME_REQS:-${PROJECT_ROOT}/third_party/requirements_verl_runtime.txt}
SFT_INSTALL_VERL_RUNTIME=${SFT_INSTALL_VERL_RUNTIME:-true}

SFT_ENV=${SFT_ENV:-sciworld}
case "$SFT_ENV" in
    sciworld|babyai|alfworld)
        ;;
    *)
        echo "错误: 无效的 SFT_ENV='$SFT_ENV'，支持: sciworld, babyai, alfworld"
        exit 1
        ;;
esac

nproc_per_node=${NPROC_PER_NODE:-2}
save_path=${SAVE_PATH:-}
model_path=${MODEL_PATH:-/Data/public/Qwen3-1.7B}
gpu_ids=${CUDA_VISIBLE_DEVICES:-0,1}
experiment_name_override=${EXPERIMENT_NAME:-}

SFT_TRAIN_BATCH_SIZE=${SFT_TRAIN_BATCH_SIZE:-64}
# fsdp_sft_trainer consumes only data.micro_batch_size_per_gpu.
SFT_MICRO_BATCH_SIZE_PER_GPU=${SFT_MICRO_BATCH_SIZE_PER_GPU:-2}
SFT_GRADIENT_ACCUMULATION_STEPS=${SFT_GRADIENT_ACCUMULATION_STEPS:-}
SFT_TRUNCATION=${SFT_TRUNCATION:-error}
SFT_TOTAL_EPOCHS=${SFT_TOTAL_EPOCHS:-60}
SFT_SAVE_FREQ=${SFT_SAVE_FREQ:-500}
SFT_TEST_FREQ=${SFT_TEST_FREQ:-1000000}
SFT_MAX_CKPT_TO_KEEP=${SFT_MAX_CKPT_TO_KEEP:-}
SFT_RESUME_MODE=${SFT_RESUME_MODE:-disable}
SFT_MODEL_DTYPE=${SFT_MODEL_DTYPE:-fp32}
SFT_ULYSSES_SEQUENCE_PARALLEL_SIZE=${SFT_ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}
SFT_USE_REMOVE_PADDING=${SFT_USE_REMOVE_PADDING:-true}
SFT_PAD_MODE=${SFT_PAD_MODE:-right}
SFT_MESSAGES_KEY=${SFT_MESSAGES_KEY:-messages}
SFT_MASTER_PORT=${SFT_MASTER_PORT:-29503}

case "$SFT_ENV" in
    sciworld)
        DEFAULT_DATASET_NAME=success_only_max4096
        DEFAULT_MAX_LENGTH=4096
        DEFAULT_ENABLE_THINKING=false
        ;;
    babyai)
        DEFAULT_DATASET_NAME=success_only
        DEFAULT_MAX_LENGTH=8192
        DEFAULT_ENABLE_THINKING=
        ;;
    alfworld)
        DEFAULT_DATASET_NAME=success_only
        DEFAULT_MAX_LENGTH=12288
        DEFAULT_ENABLE_THINKING=
        ;;
esac

DATASET_NAME=${DATASET_NAME:-$DEFAULT_DATASET_NAME}
SFT_MAX_LENGTH=${SFT_MAX_LENGTH:-$DEFAULT_MAX_LENGTH}
if [ -z "${SFT_DEFAULT_ENABLE_THINKING+x}" ]; then
    if [ "$SFT_ENV" = "sciworld" ] && [[ "$DATASET_NAME" == think_inside_* ]]; then
        SFT_DEFAULT_ENABLE_THINKING=true
    else
        SFT_DEFAULT_ENABLE_THINKING=$DEFAULT_ENABLE_THINKING
    fi
fi

declare -A DATASET_PATHS
case "$SFT_ENV" in
    sciworld)
        DATASET_PATHS[all_valid]="${SFT_DATA_ROOT}/sciworld/sciworld_all_valid.parquet"
        DATASET_PATHS[success_only]="${SFT_DATA_ROOT}/sciworld/sciworld_success_only.parquet"
        DATASET_PATHS[success_only_max4096]="${SFT_DATA_ROOT}/sciworld/sciworld_success_only_max4096.parquet"
        DATASET_PATHS[mixed]="${SFT_DATA_ROOT}/sciworld/sciworld_mixed.parquet"
        DATASET_PATHS[mixed_all_fail100]="${SFT_DATA_ROOT}/sciworld/sciworld_mixed_all_fail100.parquet"
        DATASET_PATHS[think_inside_all_valid]="${SFT_DATA_ROOT}/sciworld/sciworld_think_inside_all_valid.parquet"
        DATASET_PATHS[think_inside_success_only]="${SFT_DATA_ROOT}/sciworld/sciworld_think_inside_success_only.parquet"
        DATASET_PATHS[think_inside_success_only_max4096]="${SFT_DATA_ROOT}/sciworld/sciworld_think_inside_success_only_max4096.parquet"
        DATASET_PATHS[think_inside_mixed]="${SFT_DATA_ROOT}/sciworld/sciworld_think_inside_mixed.parquet"
        DATASET_PATHS[think_inside_mixed_all_fail100]="${SFT_DATA_ROOT}/sciworld/sciworld_think_inside_mixed_all_fail100.parquet"
        ;;
    babyai)
        DATASET_PATHS[all_valid]="${SFT_DATA_ROOT}/babyai/babyai_all_valid.parquet"
        DATASET_PATHS[success_only]="${SFT_DATA_ROOT}/babyai/babyai_success_only.parquet"
        DATASET_PATHS[success_only_max4096]="${SFT_DATA_ROOT}/babyai/babyai_success_only_max4096.parquet"
        DATASET_PATHS[mixed]="${SFT_DATA_ROOT}/babyai/babyai_mixed.parquet"
        ;;
    alfworld)
        DATASET_PATHS[all_valid]="${SFT_DATA_ROOT}/alfworld/alfworld_all_valid.parquet"
        DATASET_PATHS[success_only]="${SFT_DATA_ROOT}/alfworld/alfworld_success_only.parquet"
        DATASET_PATHS[success_only_max4096]="${SFT_DATA_ROOT}/alfworld/alfworld_success_only_max4096.parquet"
        DATASET_PATHS[mixed]="${SFT_DATA_ROOT}/alfworld/alfworld_mixed.parquet"
        ;;
esac

if [[ -z "${DATASET_PATHS[$DATASET_NAME]}" ]]; then
    echo "错误: 无效的 DATASET_NAME='$DATASET_NAME'"
    echo "支持的选项: ${!DATASET_PATHS[@]}"
    for key in "${!DATASET_PATHS[@]}"; do
        echo "  - $key: ${DATASET_PATHS[$key]}"
    done
    exit 1
fi

data_path=${DATA_PATH:-${DATASET_PATHS[$DATASET_NAME]}}
experiment_name=${experiment_name_override:-"${SFT_ENV}-sft-$(basename "$model_path")-${DATASET_NAME}"}

if [ -z "$save_path" ]; then
    save_path="${SFT_OUTPUT_ROOT}/${SFT_ENV}_sft/${SFT_ENV}_${DATASET_NAME}"
fi

LOG_DIR="${SFT_OUTPUT_ROOT}/${SFT_ENV}_sft/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${DATASET_NAME}_${TIMESTAMP}.log"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$gpu_ids
fi

export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
fi

echo "训练配置:"
echo "  env: $SFT_ENV"
echo "  GPU数量: $nproc_per_node"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  保存路径: $save_path"
echo "  模型路径: $model_path"
echo "  SFT data root: $SFT_DATA_ROOT"
echo "  SFT output root: $SFT_OUTPUT_ROOT"
echo "  SFT install verl runtime: $SFT_INSTALL_VERL_RUNTIME"
echo "  verl wheel dir: $VERL_WHEEL_DIR"
echo "  verl requirements: $VERL_RUNTIME_REQS"
echo "  数据文件: $data_path"
echo "  数据集: $DATASET_NAME"
echo "  messages_key: $SFT_MESSAGES_KEY"
echo "  实验名称: $experiment_name"
echo "  SFT train_batch_size: $SFT_TRAIN_BATCH_SIZE"
echo "  SFT micro_batch_size_per_gpu: $SFT_MICRO_BATCH_SIZE_PER_GPU"
echo "  SFT max_length: $SFT_MAX_LENGTH"
echo "  SFT truncation: $SFT_TRUNCATION"
echo "  SFT epochs: $SFT_TOTAL_EPOCHS"
echo "  SFT save_freq: $SFT_SAVE_FREQ"
echo "  SFT test_freq: $SFT_TEST_FREQ"
echo "  SFT resume_mode: $SFT_RESUME_MODE"
echo "  SFT model_dtype: $SFT_MODEL_DTYPE"
echo "  SFT ulysses_sequence_parallel_size: $SFT_ULYSSES_SEQUENCE_PARALLEL_SIZE"
echo "  SFT use_remove_padding: $SFT_USE_REMOVE_PADDING"
echo "  SFT pad_mode: $SFT_PAD_MODE"
echo "  SFT default_enable_thinking: ${SFT_DEFAULT_ENABLE_THINKING:-<dataset/default>}"
echo "  日志文件: $LOG_FILE"

extra_overrides=()
if [ -n "$SFT_GRADIENT_ACCUMULATION_STEPS" ]; then
    extra_overrides+=(+data.gradient_accumulation_steps=$SFT_GRADIENT_ACCUMULATION_STEPS)
fi
if [ -n "$SFT_MAX_CKPT_TO_KEEP" ]; then
    extra_overrides+=(trainer.max_ckpt_to_keep=$SFT_MAX_CKPT_TO_KEEP)
fi
if [ -n "$SFT_RESUME_MODE" ]; then
    extra_overrides+=(trainer.resume_mode=$SFT_RESUME_MODE)
fi
if [ -n "$SFT_DEFAULT_ENABLE_THINKING" ]; then
    extra_overrides+=(+data.multiturn.default_enable_thinking=$SFT_DEFAULT_ENABLE_THINKING)
fi

if [ "$SFT_INSTALL_VERL_RUNTIME" = "true" ]; then
    echo ""
    echo "安装 verl 框架..."
    echo "  verl wheel 目录: $VERL_WHEEL_DIR"
    echo "  verl requirements: $VERL_RUNTIME_REQS"

    if [ ! -d "$VERL_WHEEL_DIR" ]; then
        echo "错误: verl wheel 目录不存在: $VERL_WHEEL_DIR"
        echo "请在开发机运行 scripts/prepare_verl_wheels_py312.sh 并重新上传代码。"
        exit 1
    fi

    if [ ! -f "$VERL_RUNTIME_REQS" ]; then
        echo "错误: verl requirements 文件不存在: $VERL_RUNTIME_REQS"
        exit 1
    fi

    VERL_WHEEL_COUNT=$(find "$VERL_WHEEL_DIR" -maxdepth 1 -name "*.whl" -type f 2>/dev/null | wc -l)
    if [ "$VERL_WHEEL_COUNT" -eq 0 ]; then
        echo "错误: verl wheel 目录为空: $VERL_WHEEL_DIR"
        echo "当前 SFT 训练也需要这些 wheel 进行离线安装。"
        echo "请在开发机运行 scripts/prepare_verl_wheels_py312.sh，"
        echo "并把 third_party/wheels_verl_py312/ 与 third_party/requirements_verl_runtime.txt 一起上传到 OSS。"
        exit 1
    fi

    if ! python3 -m pip install \
            --no-index \
            --find-links="$VERL_WHEEL_DIR" \
            -r "$VERL_RUNTIME_REQS" \
            2>&1 | tee -a "$LOG_FILE"; then
        echo "错误: verl 运行时依赖离线安装失败。"
        echo "verl wheel 目录: $VERL_WHEEL_DIR"
        echo "verl requirements: $VERL_RUNTIME_REQS"
        exit 1
    fi

    if ! python3 -m pip install --no-build-isolation -e "$VERL_ROOT" --no-deps -q \
            2>&1 | tee -a "$LOG_FILE"; then
        echo "错误: verl 安装失败"
        exit 1
    fi
    echo "verl 安装完成"
else
    echo ""
    echo "跳过 verl 运行时依赖安装（SFT_INSTALL_VERL_RUNTIME=$SFT_INSTALL_VERL_RUNTIME）"
fi

cd "$VERL_ROOT"
torchrun --standalone --nnodes=1 --nproc_per_node="$nproc_per_node" \
    --master_port="$SFT_MASTER_PORT" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$data_path" \
    data.val_files="$data_path" \
    data.train_batch_size="$SFT_TRAIN_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$SFT_MICRO_BATCH_SIZE_PER_GPU" \
    "${extra_overrides[@]}" \
    data.max_length="$SFT_MAX_LENGTH" \
    data.truncation="$SFT_TRUNCATION" \
    +data.pad_mode="$SFT_PAD_MODE" \
    data.multiturn.enable=true \
    data.multiturn.messages_key="$SFT_MESSAGES_KEY" \
    model.partial_pretrain="$model_path" \
    model.enable_gradient_checkpointing=true \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    model.fsdp_config.model_dtype="$SFT_MODEL_DTYPE" \
    optim.lr=1e-5 \
    optim.betas='[0.9,0.95]' \
    optim.weight_decay=0.01 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir="$save_path" \
    trainer.project_name="${SFT_ENV}-sft" \
    trainer.experiment_name="$experiment_name" \
    trainer.logger='["console","wandb"]' \
    trainer.seed=42 \
    trainer.total_epochs="$SFT_TOTAL_EPOCHS" \
    trainer.save_freq="$SFT_SAVE_FREQ" \
    trainer.test_freq="$SFT_TEST_FREQ" \
    trainer.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.checkpoint.load_contents='["model","optimizer","extra"]' \
    ulysses_sequence_parallel_size="$SFT_ULYSSES_SEQUENCE_PARALLEL_SIZE" \
    use_remove_padding="$SFT_USE_REMOVE_PADDING" 2>&1 | tee -a "$LOG_FILE"

exit ${PIPESTATUS[0]}

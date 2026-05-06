#!/bin/bash
# ============================================================
# H200 集群运行脚本 - 通过 Arena 提交训练任务
# ============================================================
# 使用方式:
#   1. 先修改 config_h200.sh 中的 USER_PINYIN
#   2. 上传代码到 OSS: oss-submit.sh --prepare
#   3. 提交训练任务: oss-submit.sh --train textcraft_grpo
# ============================================================

set -e

# -------------------- 加载配置 --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_h200.sh"

# -------------------- 默认参数 --------------------
MODE=${1:-"--train"}
EXP_NAME=${2:-"textcraft_grpo"}
GPU_COUNT=${GPU_COUNT:-2}

# -------------------- 参数解析 --------------------
function show_help() {
    echo "用法: oss-submit.sh [选项] [实验名称]"
    echo ""
    echo "选项:"
    echo "  --prepare         准备阶段：上传代码到 OSS"
    echo "  --train <exp>     训练阶段：提交训练任务"
    echo "  --help            显示帮助"
    echo ""
    echo "实验名称:"
    echo "  textcraft_grpo            TextCraft 标准 GRPO"
    echo "  textcraft_grpo_validated  TextCraft Prefix-GRPO 主实验"
    echo "  textcraft_grpo_mis        TextCraft GRPO + MIS"
    echo "  textcraft_grpo_tis        TextCraft GRPO + TIS"
    echo "  babyai_grpo               BabyAI GRPO"
    echo "  webshop_grpo              WebShop GRPO"
    echo ""
    echo "示例:"
    echo "  oss-submit.sh --prepare                    # 上传代码"
    echo "  oss-submit.sh --train textcraft_grpo       # 训练 TextCraft"
    echo "  oss-submit.sh --train textcraft_grpo_validated  # 训练 TextCraft Prefix-GRPO 主实验"
    echo "  GPU_COUNT=4 oss-submit.sh --train babyai_grpo  # 4卡训练 BabyAI"
}

# -------------------- 上传代码到 OSS --------------------
function prepare() {
    echo "============================================"
    echo "  准备阶段：上传代码到 OSS"
    echo "============================================"

    if [[ "$USER_PINYIN" == "[请修改为你的姓名拼音]" ]]; then
        echo "错误: 请先修改 config_h200.sh 中的 USER_PINYIN"
        exit 1
    fi

    OSS_PATH="oss://${OSS_BUCKET}/${USER_PINYIN}/h200_grpo"
    echo "目标 OSS 路径: ${OSS_PATH}"
    echo ""
    echo "请手动上传代码到 OSS:"
    echo "  方式1: 使用 ossutil"
    echo "    ossutil cp -r ${SCRIPT_DIR}/ oss://${OSS_BUCKET}/${USER_PINYIN}/h200_grpo/"
    echo ""
    echo "上传完成后，请确保模型权重也在 OSS 上:"
    echo "  oss://${OSS_BUCKET}/${USER_PINYIN}/models/Qwen3-1.7B/"
    echo ""
    echo "提示: 模型权重也需要单独上传"
}

# -------------------- 训练阶段 --------------------
function train() {
    echo "============================================"
    echo "  训练阶段：提交 ${EXP_NAME} 任务"
    echo "============================================"

    if [[ "$USER_PINYIN" == "[请修改为你的姓名拼音]" ]]; then
        echo "错误: 请先修改 config_h200.sh 中的 USER_PINYIN"
        exit 1
    fi

    case $EXP_NAME in
        textcraft_grpo)
            SCRIPT="run_textcraft_grpo_train.sh"
            DATA_FILE="${DATA_ROOT}/textcraft/train.parquet"
            ;;
        textcraft_grpo_validated)
            SCRIPT="run_textcraft_grpo_validated.sh"
            DATA_FILE="${PROJECT_ROOT}/data/textcraft/textcraft_prefix_main_train_step200.audited.parquet"
            ;;
        textcraft_grpo_mis)
            SCRIPT="run_textcraft_grpo_mis.sh"
            DATA_FILE="${DATA_ROOT}/textcraft/train.parquet"
            ;;
        textcraft_grpo_tis)
            SCRIPT="run_textcraft_grpo_tis.sh"
            DATA_FILE="${DATA_ROOT}/textcraft/train.parquet"
            ;;
        babyai_grpo)
            SCRIPT="run_babyai_grpo_train.sh"
            DATA_FILE="${DATA_ROOT}/babyai/train.parquet"
            ;;
        webshop_grpo)
            SCRIPT="run_webshop_grpo_train.sh"
            DATA_FILE="${DATA_ROOT}/webshop/train.parquet"
            ;;
        *)
            echo "错误: 未知实验 ${EXP_NAME}"
            show_help
            exit 1
            ;;
    esac

    JOB_NAME="jiaotongdamoxing-${USER_PINYIN}-${EXP_NAME}"

    echo "实验: ${EXP_NAME}"
    echo "训练脚本: ${SCRIPT}"
    echo "数据文件: ${DATA_FILE}"
    echo "GPU 数量: ${GPU_COUNT}"
    echo "任务名称: ${JOB_NAME}"
    echo ""

    # 生成 arena submit 命令
    cat << EOF
============================================
Arena 提交命令（可直接复制执行）
============================================

arena submit pytorch \\
    --name="${JOB_NAME}" \\
    --gpus=${GPU_COUNT} \\
    --image="${TRAINING_IMAGE}" \\
    --data=dzlvpc:/oss-pvc \\
    --working-dir="${PROJECT_ROOT}" \\
    --toleration=all \\
    --env="USER_PINYIN=${USER_PINYIN}" \\
    --env="MODEL_PATH=${MODEL_ROOT}/Qwen3-1.7B" \\
    --env="DATA_PATH=${DATA_FILE}" \\
    --env="OUTPUT_DIR=${OUTPUT_ROOT}/${EXP_NAME}" \\
    --env="NUM_GPUS=${GPU_COUNT}" \\
    --env="OSS_MOUNT_ROOT=${OSS_MOUNT_ROOT}" \\
    --env="PROJECT_ROOT=${PROJECT_ROOT}" \\
    --env="VERL_ROOT=${VERL_ROOT}" \\
    --env="CONFIG_ROOT=${CONFIG_ROOT}" \\
    --env="DATA_ROOT=${DATA_ROOT}" \\
    --env="OUTPUT_ROOT=${OUTPUT_ROOT}" \\
    --env="MODEL_ROOT=${MODEL_ROOT}" \\
    --env="SCRIPTS_ROOT=${SCRIPTS_ROOT}" \\
    --env="TEXTCRAFT_PORT=${TEXTCRAFT_PORT}" \\
    --env="BABYAI_PORT=${BABYAI_PORT}" \\
    --env="WEBSHOP_PORT=${WEBSHOP_PORT}" \\
    -- \\
    bash ${SCRIPTS_ROOT}/train/${SCRIPT}

============================================
训练日志查看:
============================================
    arena logs ${JOB_NAME}
    arena logs ${JOB_NAME} -f

============================================
任务状态查看:
============================================
    arena list | grep ${JOB_NAME}

============================================
交互式 debug（如需要）:
============================================
    arena submit pytorch \\
        --name="${JOB_NAME}-debug" \\
        --gpus=${GPU_COUNT} \\
        --image="${TRAINING_IMAGE}" \\
        --data=dzlvpc:/oss-pvc \\
        --working-dir="${PROJECT_ROOT}" \\
        --toleration=all \\
        --env="USER_PINYIN=${USER_PINYIN}" \\
        --env="MODEL_PATH=${MODEL_ROOT}/Qwen3-1.7B" \\
        --env="DATA_PATH=${DATA_FILE}" \\
        --env="OUTPUT_DIR=${OUTPUT_ROOT}/${EXP_NAME}" \\
        --env="NUM_GPUS=${GPU_COUNT}" \\
        --env="OSS_MOUNT_ROOT=${OSS_MOUNT_ROOT}" \\
        --env="PROJECT_ROOT=${PROJECT_ROOT}" \\
        --env="VERL_ROOT=${VERL_ROOT}" \\
        --env="CONFIG_ROOT=${CONFIG_ROOT}" \\
        --env="DATA_ROOT=${DATA_ROOT}" \\
        --env="OUTPUT_ROOT=${OUTPUT_ROOT}" \\
        --env="MODEL_ROOT=${MODEL_ROOT}" \\
        --env="SCRIPTS_ROOT=${SCRIPTS_ROOT}" \\
        --env="TEXTCRAFT_PORT=${TEXTCRAFT_PORT}" \\
        --env="BABYAI_PORT=${BABYAI_PORT}" \\
        --env="WEBSHOP_PORT=${WEBSHOP_PORT}" \\
        -- \\
        bash

EOF
}

# -------------------- 主逻辑 --------------------
case $MODE in
    --prepare|-p)
        prepare
        ;;
    --train|-t)
        train
        ;;
    --help|-h)
        show_help
        ;;
    *)
        echo "错误: 未知选项 ${MODE}"
        show_help
        exit 1
        ;;
esac

# ============================================================
# Pod 内环境配置脚本
# ============================================================
# 在 Pod 启动后 source 此脚本设置环境变量
# 使用方式: source pod-env.sh
# ============================================================

# -------------------- OSS 挂载配置 --------------------
# 假设 oss-pvc 挂载到 /oss-pvc，用户目录在 /oss-pvc/{user_pinyin}
# 这个变量需要在外部设置或者根据实际情况修改
if [ -z "$USER_PINYIN" ]; then
    export USER_PINYIN=$(whoami)  # 默认使用当前用户名
fi

export OSS_MOUNT_ROOT="/oss-pvc/${USER_PINYIN}"

# -------------------- 项目配置 --------------------
# 项目代码目录
export PROJECT_ROOT="${OSS_MOUNT_ROOT}/h200_grpo"

# 模型目录
export MODEL_ROOT="${OSS_MOUNT_ROOT}/models"

# 数据目录
export DATA_ROOT="${PROJECT_ROOT}/data"

# 输出目录
export OUTPUT_ROOT="${OSS_MOUNT_ROOT}/outputs"

# verl 框架
export VERL_ROOT="${PROJECT_ROOT}/verl"

# 配置目录
export CONFIG_ROOT="${PROJECT_ROOT}/config"

# 脚本目录
export SCRIPTS_ROOT="${PROJECT_ROOT}/scripts"

# -------------------- 端口配置 --------------------
export TEXTCRAFT_PORT=36001
export BABYAI_PORT=36002
export SCIWORLD_PORT=36003
export ALFWORLD_PORT=36004
export WEBSHOP_PORT=36005

# -------------------- 验证 --------------------
echo "=========================================="
echo "Pod 环境配置"
echo "=========================================="
echo "USER_PINYIN: $USER_PINYIN"
echo "OSS_MOUNT_ROOT: $OSS_MOUNT_ROOT"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "MODEL_ROOT: $MODEL_ROOT"
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "VERL_ROOT: $VERL_ROOT"
echo "=========================================="

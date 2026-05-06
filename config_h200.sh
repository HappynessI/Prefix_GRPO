# ============================================================
# H200 集群配置 - 集中管理路径和参数
# ============================================================
# 使用方式: source config_h200.sh
# ============================================================

# -------------------- 用户信息（必须修改） --------------------
# 用户拼音全拼（小写）
export USER_PINYIN="[请修改为你的姓名拼音]"

# -------------------- OSS Bucket 配置 --------------------
export OSS_BUCKET="jiaotongdamoxing"
# Pod 内 OSS 挂载根目录
export OSS_MOUNT_ROOT="/oss-pvc/${USER_PINYIN}"

# -------------------- GPFS 配置（如需要） --------------------
export GPFS_ROOT="/gpfs/jiaotongdamoxing/${USER_PINYIN}"

# -------------------- 项目目录（Pod 内） --------------------
# 项目代码、配置、脚本所在目录
export PROJECT_ROOT="${OSS_MOUNT_ROOT}/h200_grpo"

# -------------------- 模型目录 --------------------
# 模型权重放在 OSS 上，挂载后访问
export MODEL_ROOT="${OSS_MOUNT_ROOT}/models"

# -------------------- 数据目录 --------------------
# 训练数据在项目目录内
export DATA_ROOT="${PROJECT_ROOT}/data"

# -------------------- 输出目录 --------------------
# 训练输出写到 OSS，方便查看
export OUTPUT_ROOT="${OSS_MOUNT_ROOT}/outputs"

# -------------------- verl 框架 --------------------
export VERL_ROOT="${PROJECT_ROOT}/verl"

# -------------------- 配置目录 --------------------
export CONFIG_ROOT="${PROJECT_ROOT}/config"

# -------------------- 脚本目录 --------------------
export SCRIPTS_ROOT="${PROJECT_ROOT}/scripts"

# -------------------- 环境配置 --------------------
export TEXTCRAFT_PORT=36001
export BABYAI_PORT=36002
export SCIWORLD_PORT=36003
export ALFWORLD_PORT=36004
export WEBSHOP_PORT=36005

# -------------------- 推荐镜像 --------------------
export TRAINING_IMAGE="cr-ee.registry.cn-hangzhou-cicore-d01.res.cncicore.com/bmcp-private/vllm:0.9.0.1-pytorch2.7-cu128-20250612"

# -------------------- 验证路径 --------------------
echo "=========================================="
echo "H200 集群配置"
echo "=========================================="
echo "USER_PINYIN: $USER_PINYIN"
echo "OSS_MOUNT_ROOT: $OSS_MOUNT_ROOT"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "MODEL_ROOT: $MODEL_ROOT"
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "VERL_ROOT: $VERL_ROOT"
echo "=========================================="
echo "端口配置:"
echo "  TEXTCRAFT: $TEXTCRAFT_PORT"
echo "  BABYAI: $BABYAI_PORT"
echo "  SCIWORLD: $SCIWORLD_PORT"
echo "  ALFWORLD: $ALFWORLD_PORT"
echo "  WEBSHOP: $WEBSHOP_PORT"
echo "=========================================="

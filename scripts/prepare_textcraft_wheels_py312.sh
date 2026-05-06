#!/bin/bash
#
# TextCraft 运行时依赖离线 wheel 打包脚本（Python 3.12 版本）
#
# 用途：在有网络的 Linux 开发机上执行，使用 Python 3.12 环境下载 TextCraft
#       运行时依赖的 wheel 文件到 third_party/wheels_py312/ 目录。
#
# 为什么需要 Python 3.12 版本：
#   H200 集群推荐镜像使用 Python 3.12。如果开发机默认 Python 版本更高（如 3.13），
#   则 wheels/py3.12 与镜像 Python 版本更对齐，建议优先使用本脚本准备 wheels。
#
# 使用方式：
#   bash scripts/prepare_textcraft_wheels_py312.sh
#
# 该脚本可重复执行。
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEEL_DIR="${PROJECT_ROOT}/third_party/wheels_py312"
REQS_FILE="${PROJECT_ROOT}/third_party/requirements_textcraft_runtime.txt"

# Python 3.12 解释器（由 conda 环境 py312_wheels 提供）
PYTHON312="/home/wyh/miniconda3/envs/py312_wheels/bin/python3.12"

echo "============================================"
echo "TextCraft 离线 wheel 打包（Python 3.12）"
echo "============================================"
echo "wheel 目录: ${WHEEL_DIR}"
echo "requirements: ${REQS_FILE}"
echo "Python 解释器: ${PYTHON312}"

if [ ! -x "${PYTHON312}" ]; then
    echo "错误: Python 3.12 解释器不存在: ${PYTHON312}"
    echo "请先创建 conda 环境：conda create -n py312_wheels python=3.12 pip -y"
    exit 1
fi

if [ ! -f "${REQS_FILE}" ]; then
    echo "错误: requirements 文件不存在: ${REQS_FILE}"
    exit 1
fi

mkdir -p "${WHEEL_DIR}"
echo "wheel 目录已创建: ${WHEEL_DIR}"

# 清理旧 wheel，避免版本混乱
echo "清理旧 wheel 文件..."
find "${WHEEL_DIR}" -maxdepth 1 \( -name "*.whl" -o -name "*.tar.gz" \) -type f -delete

# 下载依赖及其完整传递依赖树（不使用 --no-deps）
echo ""
echo "开始下载依赖及其传递依赖（Python 3.12，请确保网络通畅）..."
"${PYTHON312}" -m pip download \
    -r "${REQS_FILE}" \
    -d "${WHEEL_DIR}"

# 下载 pip / setuptools / wheel 自身（支持 Pod 内 pip 自身可用）
echo ""
echo "补充下载 pip/setuptools/wheel..."
"${PYTHON312}" -m pip download \
    pip setuptools wheel \
    -d "${WHEEL_DIR}"

# --------------- 校验 ---------------
WHEEL_COUNT=$(find "${WHEEL_DIR}" -maxdepth 1 -name "*.whl" -type f | wc -l)

echo ""
echo "============================================"
echo "wheel 打包完成（Python 3.12）"
echo "============================================"
echo "wheel 目录: ${WHEEL_DIR}"
echo "wheel 文件总数: ${WHEEL_COUNT}"
echo ""
if [ "${WHEEL_COUNT}" -gt 0 ]; then
    echo "文件列表（前 20 个）："
    ls -lh "${WHEEL_DIR}"/*.whl 2>/dev/null | head -20
else
    echo "错误: 未发现任何 .whl 文件，wheel 目录为空。"
    echo "请检查 pip 是否正常，以及 requirements 文件是否正确。"
    exit 1
fi

echo ""
echo "============================================"
echo "wheel 打包完成（Python 3.12），下一步："
echo "  1. 确认 third_party/wheels_py312/ 中已存在实际 .whl 文件："
echo "     ls third_party/wheels_py312/*.whl | head"
echo ""
echo "  2. 将整个 h200_grpo 目录上传 OSS（请确认上传了 wheels_py312/）："
echo "     ossutil cp -r ./ oss://jiaotongdamoxing/\${USER_PINYIN}/h200_grpo/"
echo ""
echo "  3. 提交训练任务："
echo "     GPU_COUNT=2 ./oss-submit.sh --train textcraft_grpo"
echo "============================================"

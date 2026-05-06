#!/bin/bash
#
# TextCraft 运行时依赖离线 wheel 打包脚本
#
# 用途：在有网络的 Linux 开发机上执行，将 TextCraft 环境服务的运行依赖
#       下载到 third_party/wheels/ 目录，随后整个 h200_grpo 目录上传 OSS，
#       Pod 内训练脚本从本地 wheel 离线安装，不再依赖 Pod 内联网。
#
# 关键保证：使用 pip download 下载完整依赖树（不含 --no-deps），
#           确保 wheel 目录中包含所有传递依赖，足以支持 Pod 内 --no-index 安装。
#
# 前置条件：
#   - 有网络的 Linux 开发机（能访问 PyPI）
#   - pip 已可用
#
# 使用方式：
#   bash scripts/prepare_textcraft_wheels.sh
#
# 该脚本可重复执行。
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEEL_DIR="${PROJECT_ROOT}/third_party/wheels"
REQS_FILE="${PROJECT_ROOT}/third_party/requirements_textcraft_runtime.txt"

echo "============================================"
echo "TextCraft 离线 wheel 打包"
echo "============================================"
echo "wheel 目录: ${WHEEL_DIR}"
echo "requirements: ${REQS_FILE}"

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
echo "开始下载依赖及其传递依赖（请确保网络通畅）..."
python3 -m pip download \
    -r "${REQS_FILE}" \
    -d "${WHEEL_DIR}"

# 下载 pip / setuptools / wheel 自身（支持 Pod 内 pip 自身可用）
echo ""
echo "补充下载 pip/setuptools/wheel..."
python3 -m pip download \
    pip setuptools wheel \
    -d "${WHEEL_DIR}"

# --------------- 校验 ---------------
WHEEL_COUNT=$(find "${WHEEL_DIR}" -maxdepth 1 -name "*.whl" -type f | wc -l)

echo ""
echo "============================================"
echo "wheel 打包完成"
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
echo "wheel 打包完成，下一步："
echo "  1. 确认 third_party/wheels/ 中已存在实际 .whl 文件："
echo "     ls third_party/wheels/*.whl | head"
echo ""
echo "  2. 将整个 h200_grpo 目录上传 OSS："
echo "     ossutil cp -r ./ oss://jiaotongdamoxing/\${USER_PINYIN}/h200_grpo/"
echo ""
echo "  3. 提交训练任务："
echo "     GPU_COUNT=2 ./oss-submit.sh --train textcraft_grpo"
echo "============================================"

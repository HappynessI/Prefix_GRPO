#!/bin/bash
set -eo pipefail

# WebShop GRPO 训练脚本
# 注意: WebShop 环境服务暂不支持。agentenv-webshop 要求 Python >=3.8,<3.9，
# 与当前推荐镜像版本不兼容，因此本脚本标记为"暂不支持"。

echo "============================================"
echo "WebShop GRPO 训练 [暂不支持]"
echo "原因: agentenv-webshop 要求 Python >=3.8,<3.9，"
echo "      当前推荐镜像版本不满足该约束。"
echo "============================================"
exit 1

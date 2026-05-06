#!/bin/bash
# ============================================================
# 导出依赖清单
# ============================================================
# 在当前环境运行，生成 requirements.txt
# ============================================================

echo "导出 verl 环境依赖..."
source ~/miniconda3/bin/activate verl
pip freeze > verl_requirements.txt
echo "  ✓ verl_requirements.txt"

echo "导出 agentenv-base 环境依赖..."
# 如果有 agentenv-base 环境
if conda env list | grep -q "^agentenv-base "; then
    source ~/miniconda3/bin/activate agentenv-base
    pip freeze > agentenv_base_requirements.txt
    echo "  ✓ agentenv_base_requirements.txt"
else
    echo "  ⚠ agentenv-base 环境不存在，跳过"
fi

echo ""
echo "依赖导出完成"
<div align="center">

# My Experiment - verl for AgentGym

<h3>
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=Online+Reinforcement+Learning+for+LLMs;AgentGym+Environment+Integration;Built+on+ByteDance+verl+Framework" alt="Typing SVG" />
</h3>

**一个基于字节跳动 verl 框架的 AgentGym 环境在线强化学习训练平台**

<br>

[![Original verl](https://img.shields.io/badge/Based_on-ByteDance_verl-blue?style=for-the-badge)](https://github.com/volcengine/verl)
[![AgentGym](https://img.shields.io/badge/Powered_by-AgentGym-green?style=for-the-badge)](https://github.com/WooooDyy/AgentGym)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)

</div>

---

## 项目概述

本项目是基于 [ByteDance verl 框架](https://github.com/volcengine/verl) 的魔改版本，旨在整合 **AgentGym** 的 HTTP 后端服务器，实现大语言模型在多种交互式环境中的 **在线强化学习训练**。

**verl** 是字节跳动 Seed 团队开源的高效、灵活的 LLM 强化学习训练库，支持 PPO、GRPO 等多种 RL 算法。本项目在其基础上扩展了对 AgentGym 环境的原生支持，使得模型可以在 TextCraft、WebShop、AlfWorld 等多个复杂环境中进行策略学习。

### 核心目标

- **无缝集成**：将 AgentGym 的 HTTP 环境后端与 verl 的 RL 训练流程完美结合
- **模块化设计**：可扩展的 Interaction 层，轻松适配新环境
- **高效训练**：利用 verl 的 FSDP、vLLM、SGLang 等先进技术加速训练
- **完整实验流程**：从数据采样、SFT 微调到 RL 训练的全流程支持

---

## 核心贡献

### 1. AgentGym 环境适配层

创建了通用的 `AgentGymBaseInteraction` 基类及多个环境的具体实现：

```
verl/interactions/
├── agentgym_base_interaction.py   # 通用基类（HTTP 请求、错误处理等）
├── textcraft_interaction.py       # TextCraft 环境（支持 ADaPT 格式）
├── webshop_interaction.py         # WebShop 电商环境
├── alfworld_interaction.py        # AlfWorld 家居环境
├── babyai_interaction.py          # BabyAI 网格世界
├── sciworld_interaction.py        # SciWorld 科学实验
├── sqlgym_interaction.py          # SQLGym 数据库查询
└── searchqa_interaction.py        # SearchQA 问答环境
```

**关键特性**：
- 统一的环境交互接口
- 自动重试和错误恢复机制
- 支持多轮对话的 session 管理
- 灵活的 action 提取和格式化

### 2. 评估框架

支持多种推理后端的评估脚本：

- **Transformers 后端**：标准的 HuggingFace 模型加载和推理
- **vLLM 后端**：高吞吐量的推理加速（GPU 利用率优化）
- **批量评估**：自动测试多个检查点和基准模型
- **多次运行**：固定随机种子的重复实验（评估稳定性）

### 3. SFT 训练

- 多轮对话的监督微调
- 支持 ADaPT 和 ReAct 等多种 prompt 格式
- 自动数据转换和预处理
- 训练完成后自动触发评估

### 4. RL 训练

- **GRPO 算法**在 AgentGym 环境的完整实现
- Webshop 环境已完成训练测试（效果待优化）
- TextCraft 环境的训练配置已准备（待测试）
- 支持灵活的设备映射和资源配置

---

## 环境配置

### Step 1: verl 框架安装

请参考 [官方 verl 文档](https://verl.readthedocs.io/) 进行环境配置。基本步骤：

```bash
# 克隆本仓库
git clone https://github.com/HappynessI/verl-for-AgentGym.git verl-agentgym
cd verl-agentgym

# 创建 conda 环境
conda create -n verl python=3.9
conda activate verl

# 安装依赖
pip install -e .
pip install -r requirements.txt
pip install -r requirements-cuda.txt  # 如果使用 CUDA
```

### Step 2: AgentGym 后端服务器

本项目使用 **AgentGym** 提供的 HTTP 后端服务器与环境交互。

**AgentGym 项目地址**：https://github.com/WooooDyy/AgentGym

请参考 AgentGym 官方文档配置和启动环境服务器。通常需要：

```bash
# 克隆 AgentGym 仓库
git clone https://github.com/WooooDyy/AgentGym.git
cd AgentGym

# 按照官方文档安装依赖和启动服务
# 具体步骤请参考：https://github.com/WooooDyy/AgentGym#installation
```

**注意**：确保环境服务器正确启动后，才能运行本项目的训练和评估脚本。

---

## 快速开始

### A. 评估实验

#### TextCraft 环境评估（Transformers 后端）

```bash
# 单次采样
cd /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval
bash run_textcraft_eval.sh

# 多次采样（8次）
NUM_SAMPLES_PER_TASK=8 bash run_textcraft_eval.sh

# 小规模测试
MAX_SAMPLES=10 NUM_SAMPLES_PER_TASK=8 bash run_textcraft_eval.sh
```

#### TextCraft 环境评估（vLLM 后端，更快）

```bash
# 单次采样
cd /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval
bash run_textcraft_eval_vllm.sh

# 多次采样（8次）
NUM_SAMPLES_PER_TASK=8 bash run_textcraft_eval_vllm.sh

# 小规模测试
MAX_SAMPLES=10 NUM_SAMPLES_PER_TASK=8 bash run_textcraft_eval_vllm.sh
```

#### 批量评估所有模型和检查点

```bash
cd /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval
bash batch_eval_all_models.sh
```

**支持的模型**：
- Qwen3-1.7B（基准模型）
- Qwen3-4B-Instruct-2507
- Qwen3-8B
- 所有 SFT 训练的检查点

---

### B. SFT 训练

#### TextCraft 环境的监督微调

```bash
cd /Data/wyh/verl/examples/sft/multiturn
bash run_textcraft_qwen3_17b_sft.sh
```

**主要配置**：
- 基础模型：Qwen3-1.7B
- 训练数据：Gemini API 采样的 ADaPT 格式数据
- 训练轮数：20 epochs
- 自动评估：训练完成后自动运行最新检查点的评估

---

### C. RL 训练

#### Webshop 环境 GRPO 训练

```bash
cd /Data/wyh/verl/examples/sglang_multiturn/my_exp/rl
bash run_webshop_grpo_train.sh
```

**训练状态**：已测试（效果待优化）

#### TextCraft 环境 GRPO 训练

```bash
cd /Data/wyh/verl/examples/sglang_multiturn/my_exp/rl
bash run_textcraft_grpo_train.sh
```

**训练状态**：配置已完成，待测试

**配置文件**：
- Webshop：`examples/sglang_multiturn/config/webshop_grpo_train.yaml`
- TextCraft：`examples/sglang_multiturn/config/textcraft_grpo_train.yaml`

---

## 实验结果

所有实验结果保存在 `outputs/` 目录下，已同步到数据集分支。

### 结果文件链接

| 实验类型 | 路径 | 说明 |
|---------|------|------|
| **SFT 训练** | [`outputs/textcraft_sft/`](https://github.com/HappynessI/My_experiment/tree/datasets/Verl-Data/outputs/textcraft_sft) | 训练日志、检查点信息 |
| **评估结果** | [`outputs/textcraft_eval/`](https://github.com/HappynessI/My_experiment/tree/datasets/Verl-Data/outputs/textcraft_eval) | 单次评估、批量对比 |
| **GRPO 训练** | [`outputs/textcraft_grpo/`](https://github.com/HappynessI/My_experiment/tree/datasets/Verl-Data/outputs/textcraft_grpo) | TextCraft RL 训练日志 |
| **Webshop RL** | [`outputs/webshop_grpo_test/`](https://github.com/HappynessI/My_experiment/tree/datasets/Verl-Data/outputs/webshop_grpo_test) | Webshop RL 训练测试 |

### 结果目录结构

```
outputs/
├── textcraft_sft/
│   ├── logs/                    # 训练日志（带时间戳）
│   └── new_ckp/                 # 检查点（已排除，过大）
├── textcraft_eval/
│   ├── logs/                    # 评估日志
│   ├── batch_eval/              # 批量评估结果
│   ├── batch_eval_new/          # 新批量评估（多次运行）
│   └── qwen3-1.7b/              # 基准模型评估
└── textcraft_grpo/
    └── logs/                    # RL 训练日志
```

**注意**：由于模型检查点文件较大，已在 `.gitignore` 中排除。如需获取训练好的模型，请查看对应的训练日志了解保存路径。

---

## 项目结构

```
verl-agentgym/
├── verl/
│   ├── interactions/              # [核心修改] AgentGym 环境适配层
│   │   ├── agentgym_base_interaction.py
│   │   ├── textcraft_interaction.py
│   │   ├── webshop_interaction.py
│   │   └── ...                    # 更多环境
│   ├── trainer/                   # verl 原有训练器
│   └── workers/                   # verl 原有 workers
│
├── examples/
│   ├── sft/multiturn/             # SFT 训练脚本
│   │   └── run_textcraft_qwen3_17b_sft.sh
│   │
│   ├── sglang_multiturn/
│   │   ├── my_exp/                # [核心修改] 主要实验代码
│   │   │   ├── eval/              # 评估脚本集合
│   │   │   │   ├── eval_textcraft_qwen3_1.7b.py
│   │   │   │   ├── eval_textcraft_qwen3_1.7b_vllm.py
│   │   │   │   ├── batch_eval_all_models.sh
│   │   │   │   └── run_textcraft_eval*.sh
│   │   │   │
│   │   │   ├── rl/                # RL 训练脚本
│   │   │   │   ├── run_webshop_grpo_train.sh
│   │   │   │   └── run_textcraft_grpo_train.sh
│   │   │   │
│   │   │   └── scripts/           # 工具脚本
│   │   │       ├── convert_fsdp_checkpoints.sh
│   │   │       └── convert_gemini_adapt_to_sft.py
│   │   │
│   │   └── config/                # 配置文件
│   │       ├── webshop_grpo_train.yaml
│   │       └── textcraft_grpo_train.yaml
│   │
│   └── data_preprocess/           # 数据预处理脚本
│       ├── convert_agentgym_data.py
│       └── batch_convert_agentgym_data.sh
│
└── outputs/                       # 实验结果输出（见数据集分支）
```

---

## 核心技术特性

### 继承自 verl 框架的优势

- **高效训练**：支持 FSDP、Megatron-LM 等分布式训练
- **灵活推理**：集成 vLLM、SGLang 等高性能推理引擎
- **算法丰富**：PPO、GRPO、ReMax 等多种 RL 算法
- **易于扩展**：混合控制器编程模型，灵活构建训练流程

### 本项目的扩展

- **环境通用性**：统一的 AgentGym 环境交互接口
- **格式灵活性**：支持 ADaPT、ReAct 等多种 prompt 格式
- **评估完善性**：多后端、多模型、多次运行的完整评估体系
- **工程友好性**：自动化脚本、日志管理、结果组织

---

## 已知问题与未来计划

### 已知问题

- Webshop 环境的 GRPO 训练效果不理想，需要调优超参数
- TextCraft 环境的 GRPO 训练尚未完成测试
- 部分环境的依赖能正常安装（lmrlgym、tool、webarena失败了）

### 未来计划

- [ ] 完成 TextCraft 环境的 RL 训练和评估
- [ ] 扩展更多 AgentGym 环境的支持（AlfWorld、BabyAI 等）
- [ ] 优化 GRPO 算法在各环境的性能
- [ ] 支持 PPO 算法的在线训练
- [ ] 添加更多评估指标和可视化工具
- [ ] 探索多环境联合训练策略

---

## 致谢

本项目基于以下优秀开源项目构建：

- **[verl](https://github.com/volcengine/verl)**：感谢字节跳动 Seed 团队开源的高效 RL 训练框架
- **[AgentGym](https://github.com/WooooDyy/AgentGym)**：感谢提供丰富的交互式环境和 HTTP 后端服务

同时感谢所有为这些项目做出贡献的开发者们！

---

## 许可证

本项目遵循原始 verl 框架的许可证。详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

如有问题或建议，欢迎通过以下方式联系：

- GitHub Issues：https://github.com/HappynessI/My_experiment/issues
- 原始 verl 项目：https://github.com/volcengine/verl

---

<div align="center">

**如果这个项目对你有帮助，请给个 Star！**

Made with coding and coffee by [HappynessI](https://github.com/HappynessI)

</div>

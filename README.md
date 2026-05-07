<div align="center">

# Prefix-GRPO

**Replayed prefix optimization for small-model agents**

[![Method](https://img.shields.io/badge/method-Prefix--GRPO-2f6fed)](#method)
[![Environments](https://img.shields.io/badge/envs-TextCraft%20%7C%20BabyAI%20%7C%20ALFWorld-0f8b8d)](#environments)
[![Code](https://img.shields.io/badge/code-verl%20%2B%20AgentGym-6b5cff)](#repository-layout)
[![Data](https://img.shields.io/badge/data-included-success)](#data-and-results)

Code and review artifact for **From Trajectories to Prefixes: Reusing
Teacher Trajectories via Replayed Prefixes and Online Continuation**.

</div>

## Review Entry Point

For paper review, start with [`ARTIFACT.md`](ARTIFACT.md). It maps the
included datasets and metrics to the TextCraft, BabyAI, and ALFWorld
experiments.

Paper-facing result mappings and curated final metrics are in
[`PAPER_RESULTS.md`](PAPER_RESULTS.md) and
`results/paper_metrics.csv`.

This repository includes:

- Replay-validated training datasets for TextCraft main and ablation runs.
- Replay-validated training datasets for BabyAI main and ablation runs.
- The replay-validated ALFWorld main training dataset.
- Training metric CSVs for TextCraft, BabyAI, and ALFWorld.
- The modified `verl` trainer and AgentGym-style environment wrappers used by
  the experiments.

Generated checkpoints, raw local logs, and experiment tracker state are not
included.

## Setup

The artifact stores parquet datasets with Git LFS. After cloning, fetch the
dataset payloads with:

```bash
git lfs install
git lfs pull
```

Recommended runtime:

```text
Python: 3.12
CUDA/PyTorch: CUDA 12.x with torch 2.8.0+cu128
Transformers: 4.56.x
SGLang: 0.5.2
vLLM: 0.11.x
Ray: 2.52.x
Pandas: 2.3.x
PyArrow: 22.x
Hydra: 1.3.x
Matplotlib: 3.10.x
```

Install the core Python dependencies from the repository manifests:

```bash
pip install -r verl/requirements.txt
pip install -r verl/requirements_sglang.txt
pip install -e verl/
```

Environment wrappers can be installed as needed, for example:

```bash
pip install -e envs/AgentGym/agentenv-textcraft
pip install -e envs/AgentGym/agentenv-babyai
pip install -e envs/AgentGym/agentenv-alfworld
```

## Overview

Prefix-GRPO reuses teacher trajectories as more than one-shot imitation targets.
It splits a teacher rollout into replayable prefix queries, restores each
intermediate environment state by replay, and trains the student on online
continuations from those cut states.

Unlike response-only GRPO, the implementation can also optimize historical
assistant tokens inside the replayed prefix. Prefix old log-probabilities are
computed under a policy-distilled SFT checkpoint and stored as prompt-space
sidecars, which lets prefix tokens enter the same clipped policy optimization
family as continuation tokens.

## Method

```text
teacher trajectory
      |
      |  entropy-change cut selection
      v
replayable prefix queries
      |
      |  environment replay validation
      v
canonical prompt at cut state
      |
      |  online rollout
      v
Prefix-GRPO objective
  - continuation token update
  - historical assistant prefix token update
```

The main TextCraft pipeline uses assistant-token **Entropy-Change Top-3**
selection with a centered window of `w=11`, then validates candidate cuts by
environment replay before admitting them to training.

## Current Main Setting

The current TextCraft main experiment is:

```text
Dataset: Entropy-Change Top-3, replay validated
Objective: split Prefix-GRPO
Prefix advantage: cont_mean_abs
KL loss: disabled
Entropy coefficient: 0
```

The launch script sets the active defaults:

```bash
bash scripts/train/run_textcraft_grpo_validated.sh
```

Core overrides:

```text
optimize_prefix_tokens=true
prefix_loss_mode=split
prefix_advantage_mode=cont_mean_abs
use_kl_loss=false
enable_activation_offload=false
entropy_coeff=0
```

`constant` prefix advantage is still supported for ablations, but it is not the
main setting.

## Data And Results

Dataset inventory:

```text
data/DATASETS.csv
data/SCHEMA.md
data/SCHEMA_COLUMNS.csv
```

Included datasets:

```text
data/textcraft/main/main_change_top3_w11_fullflow.parquet
data/textcraft/ablations/*.parquet
data/babyai/main/main_change_top3_w11_fullflow.parquet
data/babyai/ablations/*.parquet
data/alfworld/main/main_change_top3_w11_fullflow.parquet
```

Training metric inventory:

```text
results/training_metrics_index.csv
results/paper_metrics.csv
```

Main experiment reward curves:

![Main experiment training reward curves](results/figures/main_reward_curves.png)

Raw training CSVs are stored under `results/training/`.

## Train

Main TextCraft Prefix-GRPO:

```bash
MODEL_ROOT=checkpoints \
OUTPUT_ROOT=outputs/textcraft_grpo \
DATA_PATH=data/textcraft/main/main_change_top3_w11_fullflow.parquet \
NUM_GPUS=2 \
bash scripts/train/run_textcraft_grpo_validated.sh
```

Small smoke run:

```bash
MODEL_ROOT=checkpoints \
OUTPUT_ROOT=outputs/smoke_textcraft \
DATA_PATH=data/textcraft/main/main_change_top3_w11_fullflow.parquet \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
DEBUG_MODE=1 \
DEBUG_MAX_SAMPLES=2 \
NUM_EPOCHS=1 \
bash scripts/train/run_textcraft_grpo_validated.sh
```

For smoke tests, keep checkpoint saving disabled unless a checkpoint artifact is
part of the test.

## Repository Layout

```text
Prefix_GRPO/
├── ARTIFACT.md              # Review-oriented artifact guide
├── PAPER_RESULTS.md         # Paper result to artifact mapping
├── config/                  # GRPO and environment interaction configs
├── data/                    # Included replay-validated datasets and inventory
├── envs/AgentGym/           # Environment wrappers and local servers
├── legacy_eval/             # Legacy vLLM-server evaluation utilities
├── results/                 # Included training metrics and curated paper metrics
├── scripts/
│   ├── build_data/          # Prefix data construction and validation
│   ├── train/               # Training and SFT launch scripts
│   └── utils/               # Small helper utilities
├── third_party/             # Runtime requirement manifests only
└── verl/                    # Modified verl training stack
```

## Environments

| Environment | Artifact coverage |
| --- | --- |
| TextCraft | Main dataset, ablation datasets, training metrics |
| BabyAI | Main dataset, ablation datasets, training metrics |
| ALFWorld | Main dataset; Prefix-GRPO main training metrics |
| SciWorld | Environment wrapper included; not part of the Prefix-GRPO artifact tables |
| WebShop | Wrapper included; not part of the Prefix-GRPO artifact tables |

## Notes

- Prefix annotations are in prompt-token coordinates on the canonicalized
  prompt. Prefix tokens are often in the middle of the prompt, not at the tail.
- The current `joint` objective variant shares clipping with continuation tokens
  and should not be described as strict full-trajectory PPO.
- Entropy is used for offline cut selection; `entropy_coeff=0`, so entropy is
  not an active reward term or training regularizer in the main experiment.

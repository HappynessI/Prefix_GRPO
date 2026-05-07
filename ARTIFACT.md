# Review Artifact

This document is the main entry point for reviewing the Prefix-GRPO code and
experiment artifacts. Paths are relative to the repository root.

## What Is Included

| Area | Contents |
| --- | --- |
| Code | Modified `verl` trainer, AgentGym-style environments, data builders, launch scripts |
| Data | Replay-validated Prefix-GRPO training parquet files under `data/` |
| Metrics | Training CSVs under `results/` |
| Inventory | `data/DATASETS.csv`, `data/SCHEMA.md`, `results/training_metrics_index.csv`, `results/paper_metrics.csv` |

Checkpoints and raw local experiment logs are not included.

## Dataset Map

The full dataset inventory, including row counts, file sizes, SHA256 hashes,
and prefix-sidecar field presence, is in:

```text
data/DATASETS.csv
```

Main datasets:

| Environment | Dataset |
| --- | --- |
| TextCraft | `data/textcraft/main/main_change_top3_w11_fullflow.parquet` |
| BabyAI | `data/babyai/main/main_change_top3_w11_fullflow.parquet` |
| ALFWorld | `data/alfworld/main/main_change_top3_w11_fullflow.parquet` |

Ablation datasets:

| Environment | Datasets |
| --- | --- |
| TextCraft | `data/textcraft/ablations/main_fixed_gp1_fullflow.parquet`, `main_fixed_gp2_fullflow.parquet`, `main_raw_top3_fullflow.parquet`, `main_change_top3_w5_fullflow.parquet` |
| BabyAI | `data/babyai/ablations/main_fixed_gp1_fullflow.parquet`, `main_fixed_gp2_fullflow.parquet`, `main_raw_top3_fullflow.parquet` |

All included Prefix-GRPO datasets contain the prompt-space sidecar fields used
for prefix optimization:

```text
assistant_prefix_span
prefix_mask
assistant_prefix_old_log_probs
prefix_token_count
```

## Metrics Map

Training metrics:

```text
results/training_metrics_index.csv
results/paper_metrics.csv
PAPER_RESULTS.md
results/training/textcraft/
results/training/babyai/
results/training/alfworld/
```

The index file provides a compact table of the included training metrics and
SHA256 hashes for the raw CSV files. `results/paper_metrics.csv` is a curated
mapping from paper-facing experiment groups to final metric rows.

## Main TextCraft Run

The main TextCraft Prefix-GRPO run uses:

```text
Dataset: data/textcraft/main/main_change_top3_w11_fullflow.parquet
Objective: split Prefix-GRPO
Prefix advantage: cont_mean_abs
KL loss: disabled
Entropy coefficient: 0
Launch script: scripts/train/run_textcraft_grpo_validated.sh
```

Example command:

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

## Data Construction

The main data builders are:

```text
scripts/build_data/compute_textcraft_teacher_entropy.py
scripts/build_data/build_textcraft_main_prefix_new_main_prefix.py
scripts/build_data/build_babyai_prefix_rl_change_top3.py
scripts/build_data/build_alfworld_prefix_rl_change_top3.py
```

The public artifact already includes the replay-validated outputs needed for
review. The builders are included to show how the prefix datasets were produced
from teacher trajectories and SFT old-log-probability computation.

## Scope Notes

- TextCraft is the primary environment for the main paper setting.
- BabyAI includes both main and ablation Prefix-GRPO datasets and metrics.
- ALFWorld includes the main Prefix-GRPO dataset and a partial merged
  Prefix-GRPO metric file. The available metric segments cover steps 1-501 and
  1001-1120; steps 502-1000 are not included in the current artifact.
- SciWorld and WebShop wrappers are kept because they are part of the inherited
  environment stack, but they are not part of the included Prefix-GRPO artifact
  table.
- `legacy_eval/` contains older vLLM-service evaluation utilities. Prefer the
  summarized metrics under `results/` for artifact review.

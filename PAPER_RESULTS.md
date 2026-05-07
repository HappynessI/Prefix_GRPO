# Paper Result Mapping

This file maps the curated review artifact to the paper-facing experiment
groups. The machine-readable version is:

```text
results/paper_metrics.csv
```

The CSV records the dataset path, metric CSV path, final step, final training
reward/score, TextCraft validation accuracy when available, and the prefix-loss
diagnostics used to verify that Prefix-GRPO is active.

## Main Results

| Environment | Dataset | Metrics | Status |
| --- | --- | --- | --- |
| TextCraft | `data/textcraft/main/main_change_top3_w11_fullflow.parquet` | `results/training/textcraft/training_metrics_main-change-top3.csv` | available |
| BabyAI | `data/babyai/main/main_change_top3_w11_fullflow.parquet` | `results/training/babyai/prefix_grpo/training_metrics_prefix_grpo.csv` | available |
| ALFWorld | `data/alfworld/main/main_change_top3_w11_fullflow.parquet` | `results/training/alfworld/prefix_grpo/training_metrics_prefix_grpo_merged_partial.csv` | partial |

## Ablations

| Environment | Ablation | Dataset | Metrics | Status |
| --- | --- | --- | --- | --- |
| TextCraft | fixed ratio group 1 | `data/textcraft/ablations/main_fixed_gp1_fullflow.parquet` | `results/training/textcraft/training_metrics_main-fixed-gp1.csv` | available |
| TextCraft | fixed ratio group 2 | `data/textcraft/ablations/main_fixed_gp2_fullflow.parquet` | `results/training/textcraft/training_metrics_main-fixed-gp2.csv` | available |
| TextCraft | raw entropy Top-3 | `data/textcraft/ablations/main_raw_top3_fullflow.parquet` | `results/training/textcraft/training_metrics_main-raw-top3.csv` | available |
| TextCraft | no prefix optimization | `data/textcraft/main/main_change_top3_w11_fullflow.parquet` | `results/training/textcraft/training_metrics_no_prefix_opt.csv` | available |
| TextCraft | joint objective variant | `data/textcraft/main/main_change_top3_w11_fullflow.parquet` | `results/training/textcraft/training_metrics_main-joint-gp1.csv` | available |
| BabyAI | fixed ratio group 1 | `data/babyai/ablations/main_fixed_gp1_fullflow.parquet` | `results/training/babyai/prefix_grpo_fixedgp1/training_metrics_prefix_grpo.csv` | available |
| BabyAI | fixed ratio group 2 | `data/babyai/ablations/main_fixed_gp2_fullflow.parquet` | `results/training/babyai/prefix_grpo_fixedgp2/training_metrics_prefix_grpo.csv` | available |
| BabyAI | raw entropy Top-3 | `data/babyai/ablations/main_raw_top3_fullflow.parquet` | `results/training/babyai/prefix_grpo_raw/training_metrics_prefix_grpo.csv` | available |

## Artifact-Only Metrics

TextCraft w5 is included as an artifact-only window ablation:

```text
data/textcraft/ablations/main_change_top3_w5_fullflow.parquet
results/training/textcraft/training_metrics_prefix_grpo_w5.csv
```

The current manuscript does not report w5 or w7 window ablation results.

## Partial Metrics

ALFWorld main Prefix-GRPO metrics are currently partial. The merged file:

```text
results/training/alfworld/prefix_grpo/training_metrics_prefix_grpo_merged_partial.csv
```

contains the provided segments for steps 1-501 and 1001-1120. Steps 502-1000
are not included in the current artifact.

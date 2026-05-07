# Prefix-GRPO Artifact Maintenance

## 2026-05-07 Review Artifact Packaging

- Added `ARTIFACT.md` as the review-oriented entry point.
- Reworked `README.md` to describe included datasets and metrics instead of
  referring to local machine paths.
- Added replay-validated TextCraft, BabyAI, and ALFWorld parquet datasets under
  `data/`.
- Added `data/DATASETS.csv` with row counts, SHA256 hashes, and prefix-sidecar
  field checks.
- Added training metric CSVs under `results/`.
- Added `results/training_metrics_index.csv` and
  later removed the evaluation-summary index because review metrics are tracked
  separately from H200-side evaluation summaries.
- Replaced local absolute paths in public scripts and legacy evaluation files
  with repository-relative paths or generic checkpoint placeholders.
- Replaced the vendored `verl/README.md` with an artifact-specific note to avoid
  stale private links.

## 2026-05-07 Metrics Scope Adjustment

- Removed `results/eval/` and `results/eval_metrics_index.csv`; evaluation
  summaries are not part of the public artifact payload for now.
- Added the TextCraft w5 training metrics from
  `training_metrics_prefix_grpo_w5.csv`.
- Regenerated `results/training_metrics_index.csv` with 31 training metric CSVs.

## 2026-05-07 BabyAI Ablation Metrics And TextCraft W7 Removal

- Removed `data/textcraft/ablations/main_change_top3_w7_fullflow.parquet`
  because the corresponding w7 training run will not be included.
- Regenerated `data/DATASETS.csv` with 10 datasets.
- Added BabyAI Prefix-GRPO ablation training metrics from
  `prefix_grpo_fixedgp1`, `prefix_grpo_fixedgp2`, and `prefix_grpo_raw`.
- Regenerated `results/training_metrics_index.csv` with 34 training metric CSVs.
- Marked ALFWorld main Prefix-GRPO training metrics as pending in the artifact
  guide.

## 2026-05-07 Review Mapping And Schema Docs

- Added `PAPER_RESULTS.md` and `results/paper_metrics.csv` to map paper-facing
  experiment groups to dataset and metrics files.
- Added `data/SCHEMA.md` and `data/SCHEMA_COLUMNS.csv` to document the parquet
  schema and Prefix-GRPO sidecar fields.
- Added README setup guidance for Git LFS and tested dependency versions.

## 2026-05-07 ALFWorld Partial Prefix-GRPO Metrics

- Added merged partial ALFWorld Prefix-GRPO metrics under
  `results/training/alfworld/prefix_grpo/`.
- The available segments cover steps 1-501 and 1001-1120; steps 502-1000 are
  not included.
- Regenerated `results/training_metrics_index.csv` and
  `results/paper_metrics.csv`.

## 2026-05-07 ALFWorld Recovered Prefix-GRPO Metrics

- Recovered the complete ALFWorld Prefix-GRPO training metrics from structured
  training metric lines.
- Replaced the partial merged file with
  `results/training/alfworld/prefix_grpo/training_metrics_prefix_grpo.csv`.
- The recovered CSV covers steps 1-1120 and keeps only sanitized metric columns;
  raw local logs remain excluded from the public artifact.
- Regenerated `results/training_metrics_index.csv` and
  `results/paper_metrics.csv`.

## 2026-05-07 Main Reward Curve Figure

- Added `scripts/utils/plot_main_reward_curves.py` to plot the three
  paper-main training reward curves from `results/paper_metrics.csv`.
- Added `results/figures/main_reward_curves.png` and referenced it from
  `README.md`.
- Added the matplotlib version used for reproducing the figure to the README
  dependency list.

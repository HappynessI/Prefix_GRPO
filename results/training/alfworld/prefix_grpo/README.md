# ALFWorld Prefix-GRPO Metrics

This directory contains a merged partial ALFWorld Prefix-GRPO training metric file built from:

- `training_metrics_prefix_grpo.csv`
- `training_metrics_prefix_grpo_last114.csv`

The merged file is:

```text
training_metrics_prefix_grpo_merged_partial.csv
```

Rows: 621
Step range: 1-1120

Missing step range in provided segments: 502-1000.

The file includes Prefix-GRPO diagnostic columns such as `actor_prefix_loss`, `actor_prefix_token_count`, and `actor_prefix_advantage_abs_mean`, but it should be treated as partial until the missing middle segment is available.

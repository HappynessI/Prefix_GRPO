# Prefix-GRPO

Code for **From Trajectories to Prefixes: Reusing Teacher Trajectories via
Replayed Prefixes and Online Continuation**.

Prefix-GRPO trains small multi-turn agents from teacher trajectories by turning
one long demonstration into multiple replay-aligned prefix queries. Each query
replays a teacher prefix to recover an intermediate environment state, then
lets the student continue online and optimize task reward. In addition to the
standard continuation tokens, Prefix-GRPO can optimize historical assistant
tokens inside the replayed prefix by using SFT-computed old log-probabilities in
prompt space.

The current release is synchronized from the H200 experiment codebase at
`/Data/wyh/h200_grpo_from_oss`. Local maintenance/debug notes and generated
datasets are intentionally not included.

## What Is Included

- `verl/`: the modified verl training stack used by Prefix-GRPO.
- `config/`: GRPO and environment interaction configs.
- `envs/AgentGym/`: local AgentGym environment wrappers for TextCraft, BabyAI,
  SciWorld, ALFWorld, and WebShop.
- `scripts/build_data/`: offline dataset construction, replay validation,
  canonicalization, entropy scoring, and old-logprob sidecar scripts.
- `scripts/train/`: H200-oriented training and SFT launch scripts.
- `legacy_eval/`: legacy vLLM-server evaluation utilities.
- `third_party/requirements_*.txt`: runtime dependency manifests. Large offline
  wheel directories are excluded from Git.

## Current Main Setting

The TextCraft main experiment uses the replay-validated Entropy-Change Top-3
dataset:

```text
/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/replay_validated/main_change_top3_w11_fullflow.parquet
```

The active Prefix-GRPO defaults are set by
`scripts/train/run_textcraft_grpo_validated.sh`:

```text
optimize_prefix_tokens=true
prefix_loss_mode=split
prefix_advantage_mode=cont_mean_abs
use_kl_loss=false
enable_activation_offload=false
entropy_coeff=0
```

`constant` prefix advantage remains available as an explicit ablation, but it is
not the current main setting.

## Data Layout

Training data is not committed to this repository. For the default TextCraft
script, either copy or symlink the official parquet to:

```text
data/textcraft/replay_validated/main_change_top3_w11_fullflow.parquet
```

or pass it explicitly:

```bash
DATA_PATH=/path/to/main_change_top3_w11_fullflow.parquet \
  bash scripts/train/run_textcraft_grpo_validated.sh
```

## Build Data

Data construction scripts are separated from training launch scripts:

```text
scripts/build_data/
  compute_textcraft_teacher_entropy.py
  build_textcraft_main_prefix_new_main_prefix.py
  build_textcraft_teacher_demo_rows.py
  build_babyai_prefix_rl_change_top3.py
  build_alfworld_prefix_rl_change_top3.py
```

The TextCraft pipeline follows these stages:

1. Compute teacher-forcing entropy with the policy-distilled SFT checkpoint.
2. Select replayable cut states with assistant-token Entropy-Change Top-3.
3. Replay prefix actions in the environment and keep validated states.
4. Canonicalize prompts as `system + prefix history + cut-state observation`.
5. Store prompt-space prefix sidecars:
   `assistant_prefix_span`, `prefix_mask`, and
   `assistant_prefix_old_log_probs`.

Example:

```bash
python scripts/build_data/build_textcraft_main_prefix_new_main_prefix.py \
  --output-root /Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix
```

BabyAI and ALFWorld prefix data builders live in the same directory. Their tmux
wrappers remain in `scripts/train/` because they launch environment servers and
coordinate GPU-side old-logprob computation.

## Train

The H200 pod path should use the real OSS/PVC directory name:

```text
/oss-pvc/zhk_wyh/h200_grpo
```

not the local development directory name `h200_grpo_from_oss`.

Main TextCraft Prefix-GRPO:

```bash
MODEL_ROOT=/oss-pvc/zhk_wyh/models \
OUTPUT_ROOT=/oss-pvc/zhk_wyh/outputs \
NUM_GPUS=2 \
bash scripts/train/run_textcraft_grpo_validated.sh
```

Useful smoke-test override:

```bash
MODEL_ROOT=/path/to/models \
OUTPUT_ROOT=/tmp/prefix_grpo_outputs \
DATA_PATH=/path/to/main_change_top3_w11_fullflow.parquet \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
DEBUG_MODE=1 \
DEBUG_MAX_SAMPLES=2 \
NUM_EPOCHS=1 \
bash scripts/train/run_textcraft_grpo_validated.sh
```

Do not save checkpoints during smoke tests unless you explicitly need them.

## Notes

- The repository is code-first. Generated parquet/jsonl/csv artifacts,
  checkpoints, logs, offline wheels, and local run outputs are ignored.
- Prefix annotations are prompt-space annotations on the canonicalized prompt;
  prefix tokens are often in the middle of the prompt, not at the tail.
- The current `joint` objective variant shares clipping with continuation tokens
  and should not be interpreted as strict full-trajectory PPO.

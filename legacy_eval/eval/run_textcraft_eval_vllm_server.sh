#!/bin/bash
set -e

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}

# 配置参数
VLLM_SERVER_URL=${VLLM_SERVER_URL:-"http://127.0.0.1:8000"}
TEXTCRAFT_SERVER=${TEXTCRAFT_SERVER:-"http://127.0.0.1:36001"}
MODEL_NAME=${MODEL_NAME:-"qwen3"}
DATA_PATH=${DATA_PATH:-"${PROJECT_ROOT}/data/textcraft/test.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/textcraft_eval"}

# 环境变量覆盖
MAX_SAMPLES=${MAX_SAMPLES:--1}          # -1 means all samples
NUM_SAMPLES_PER_TASK=${NUM_SAMPLES_PER_TASK:-8}  # Keep pass@1~8 rollout-style evaluation
CONCURRENCY=${CONCURRENCY:-4}           # Conservative async episode concurrency
MAX_ROUNDS=${MAX_ROUNDS:-30}            # Official eval uses 30 interaction rounds
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}   # Official eval per-turn generation cap
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-10240}  # Match current SFT eval context budget
TEMPERATURE=${TEMPERATURE:-1.0}         # Official eval temperature
TOP_P=${TOP_P:-1.0}                     # Official eval does not override top_p
REQUEST_RETRIES=${REQUEST_RETRIES:-5}   # Retry transient/empty vLLM responses
RETRY_BACKOFF_SECONDS=${RETRY_BACKOFF_SECONDS:-1.0}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$OUTPUT_DIR/eval.log"}




# 运行评估
python "${PROJECT_ROOT}/legacy_eval/eval/eval_textcraft_vllm_server.py" \
  --vllm_server_url "$VLLM_SERVER_URL" \
  --textcraft_server "$TEXTCRAFT_SERVER" \
  --model_name "$MODEL_NAME" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_rounds "$MAX_ROUNDS" \
  --max_samples "$MAX_SAMPLES" \
  --num_samples_per_task "$NUM_SAMPLES_PER_TASK" \
  --concurrency "$CONCURRENCY" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_context_tokens "$MAX_CONTEXT_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --request_retries "$REQUEST_RETRIES" \
  --retry_backoff_seconds "$RETRY_BACKOFF_SECONDS" \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "评估完成! 结果已保存至:" | tee -a "$LOG_FILE"
grep -A 2 "Output file" "$LOG_FILE" | tail -n 2 | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

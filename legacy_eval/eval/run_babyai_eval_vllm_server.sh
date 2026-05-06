#!/bin/bash
set -e

# 配置参数
VLLM_SERVER_URL="http://localhost:8000"
MODEL_NAME=${MODEL_NAME:-"qwen3"}          # vLLM中注册的模型名称
BABYAI_SERVER="http://127.0.0.1:36005"
DATA_PATH="/Data/wyh/datasets/Verl-Data/eval/babyai/test.parquet"
OUTPUT_DIR="/Data/wyh/datasets/Verl-Data/outputs/babyai_eval"

# 环境变量覆盖
MAX_SAMPLES=${MAX_SAMPLES:--1}          # -1 means all samples
NUM_SAMPLES_PER_TASK=${NUM_SAMPLES_PER_TASK:-8}  # Number of samples per task
CONCURRENCY=${CONCURRENCY:-8}          # Concurrent requests to vLLM (降低避免超时)
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}   # Max tokens per generation
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-10240}  # vLLM context window budget
TEMPERATURE=${TEMPERATURE:-1.0}         # Sampling temperature
TOP_P=${TOP_P:-1.0}                     # Top-p sampling
MAX_ROUNDS=${MAX_ROUNDS:-30}            # Max interaction rounds

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_service_${TIMESTAMP}.log"

# 运行评估
python /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/eval_babyai_vllm_server.py \
  --vllm_server_url "$VLLM_SERVER_URL" \
  --model_name "$MODEL_NAME" \
  --babyai_server "$BABYAI_SERVER" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --num_samples_per_task "$NUM_SAMPLES_PER_TASK" \
  --concurrency "$CONCURRENCY" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_context_tokens "$MAX_CONTEXT_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_rounds "$MAX_ROUNDS" \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "评估完成! 结果已保存至:" | tee -a "$LOG_FILE"
grep -A 2 "Output file" "$LOG_FILE" | tail -n 2 | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

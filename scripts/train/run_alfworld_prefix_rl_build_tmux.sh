#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/Data/wyh/h200_grpo_from_oss"}
PYTHON_BIN=${PYTHON_BIN:-"/Data/wyh/conda_envs/verl/bin/python"}
ALFWORLD_BIN=${ALFWORLD_BIN:-"/Data/wyh/conda_envs/agentenv-alfworld/bin/alfworld"}
SCRIPT_PATH=${SCRIPT_PATH:-"${PROJECT_ROOT}/scripts/build_data/build_alfworld_prefix_rl_change_top3.py"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/Data/wyh/datasets/Verl-Data/train/alfworld/prefix-rl"}
SERVER_PORT=${SERVER_PORT:-36016}
ALFWORLD_SERVER_COUNT=${ALFWORLD_SERVER_COUNT:-1}
SERVER_URL=${SERVER_URL:-}
ALFWORLD_AGENTENV_ROOT=${ALFWORLD_AGENTENV_ROOT:-"${PROJECT_ROOT}/envs/AgentGym/agentenv-alfworld"}
GPU0=${GPU0:-0}
GPU1=${GPU1:-7}
STAGE1_BATCH_PROGRESS=${STAGE1_BATCH_PROGRESS:-50}
BUILD_PROGRESS=${BUILD_PROGRESS:-100}
REPLAY_CONCURRENCY=${REPLAY_CONCURRENCY:-32}
SIDECAR_BATCH_SIZE=${SIDECAR_BATCH_SIZE:-2}
MAX_BATCH_PROMPT_TOKENS=${MAX_BATCH_PROMPT_TOKENS:-2400}

LOG_DIR="${OUTPUT_ROOT}/logs"
STAGE1_DIR="${OUTPUT_ROOT}/stage1_forward"
mkdir -p "${LOG_DIR}" "${STAGE1_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/build_${TIMESTAMP}.log"
SHARD0_LOG="${LOG_DIR}/stage1_shard0_${TIMESTAMP}.log"
SHARD1_LOG="${LOG_DIR}/stage1_shard1_${TIMESTAMP}.log"
BUILD_LOG="${LOG_DIR}/build_stage_${TIMESTAMP}.log"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${ALFWORLD_AGENTENV_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "AlfWorld prefix-RL build started at ${TIMESTAMP}" | tee -a "${MAIN_LOG}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}" | tee -a "${MAIN_LOG}"
echo "GPU0=${GPU0}, GPU1=${GPU1}" | tee -a "${MAIN_LOG}"

SERVER_PIDS=()
SERVER_URLS=()
for offset in $(seq 0 $((ALFWORLD_SERVER_COUNT - 1))); do
    port=$((SERVER_PORT + offset))
    SERVER_URLS+=("http://127.0.0.1:${port}")
done
if [ -z "${SERVER_URL}" ]; then
    SERVER_URL=$(IFS=,; echo "${SERVER_URLS[*]}")
else
    IFS=',' read -r -a SERVER_URLS <<< "${SERVER_URL}"
fi
echo "SERVER_URL=${SERVER_URL}" | tee -a "${MAIN_LOG}"
echo "ALFWORLD_SERVER_COUNT=${#SERVER_URLS[@]}" | tee -a "${MAIN_LOG}"

cleanup() {
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

for server_url in "${SERVER_URLS[@]}"; do
    port="${server_url##*:}"
    SERVER_LOG="${LOG_DIR}/alfworld_server_${port}_${TIMESTAMP}.log"
    "${ALFWORLD_BIN}" --host 127.0.0.1 --port "${port}" > "${SERVER_LOG}" 2>&1 &
    server_pid="$!"
    SERVER_PIDS+=("${server_pid}")
    READY=false
    for _ in $(seq 1 60); do
        if curl -fsS "${server_url}/" >/dev/null 2>&1; then
            READY=true
            break
        fi
        sleep 2
    done
    if [ "${READY}" != "true" ]; then
        echo "AlfWorld server failed to become ready. See ${SERVER_LOG}" | tee -a "${MAIN_LOG}"
        exit 1
    fi
    echo "AlfWorld server ready, pid=${server_pid}, url=${server_url}, log=${SERVER_LOG}" | tee -a "${MAIN_LOG}"
done

CUDA_VISIBLE_DEVICES="${GPU0}" "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --stage stage1 \
    --output-root "${OUTPUT_ROOT}" \
    --num-shards 2 \
    --shard-index 0 \
    --stage1-output-path "${STAGE1_DIR}/alfworld_sft_step930_oldlogprob_entropy.shard00-of-02.parquet" \
    --progress-every "${STAGE1_BATCH_PROGRESS}" \
    > "${SHARD0_LOG}" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --stage stage1 \
    --output-root "${OUTPUT_ROOT}" \
    --num-shards 2 \
    --shard-index 1 \
    --stage1-output-path "${STAGE1_DIR}/alfworld_sft_step930_oldlogprob_entropy.shard01-of-02.parquet" \
    --progress-every "${STAGE1_BATCH_PROGRESS}" \
    > "${SHARD1_LOG}" 2>&1 &
PID1=$!

echo "Stage1 shard processes started: ${PID0}, ${PID1}" | tee -a "${MAIN_LOG}"
wait "${PID0}"
wait "${PID1}"
echo "Stage1 shards completed." | tee -a "${MAIN_LOG}"

CUDA_VISIBLE_DEVICES="${GPU1}" "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --stage build \
    --output-root "${OUTPUT_ROOT}" \
    --server "${SERVER_URL}" \
    --replay-concurrency "${REPLAY_CONCURRENCY}" \
    --batch-size "${SIDECAR_BATCH_SIZE}" \
    --max-batch-prompt-tokens "${MAX_BATCH_PROMPT_TOKENS}" \
    --progress-every "${BUILD_PROGRESS}" \
    > "${BUILD_LOG}" 2>&1

echo "Build stage completed." | tee -a "${MAIN_LOG}"
echo "Main log: ${MAIN_LOG}" | tee -a "${MAIN_LOG}"
echo "Shard0 log: ${SHARD0_LOG}" | tee -a "${MAIN_LOG}"
echo "Shard1 log: ${SHARD1_LOG}" | tee -a "${MAIN_LOG}"
echo "Build log: ${BUILD_LOG}" | tee -a "${MAIN_LOG}"

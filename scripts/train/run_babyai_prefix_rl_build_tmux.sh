#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/Data/wyh/h200_grpo_from_oss"}
PYTHON_BIN=${PYTHON_BIN:-"/Data/wyh/conda_envs/verl/bin/python"}
BABYAI_BIN=${BABYAI_BIN:-"/Data/wyh/conda_envs/agentenv-babyai/bin/babyai"}
SCRIPT_PATH=${SCRIPT_PATH:-"${PROJECT_ROOT}/scripts/build_data/build_babyai_prefix_rl_change_top3.py"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/Data/wyh/datasets/Verl-Data/train/babyai/prefix-rl"}
DATASETS=${DATASETS:-"main_change_top3_w11_fullflow"}
SERVER_PORT=${SERVER_PORT:-36015}
SERVER_URL=${SERVER_URL:-"http://127.0.0.1:${SERVER_PORT}"}
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
SERVER_LOG="${LOG_DIR}/babyai_server_${SERVER_PORT}_${TIMESTAMP}.log"
SHARD0_LOG="${LOG_DIR}/stage1_shard0_${TIMESTAMP}.log"
SHARD1_LOG="${LOG_DIR}/stage1_shard1_${TIMESTAMP}.log"
BUILD_LOG="${LOG_DIR}/build_stage_${TIMESTAMP}.log"

cd "${PROJECT_ROOT}"

echo "BabyAI prefix-RL build started at ${TIMESTAMP}" | tee -a "${MAIN_LOG}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}" | tee -a "${MAIN_LOG}"
echo "DATASETS=${DATASETS}" | tee -a "${MAIN_LOG}"
echo "SERVER_URL=${SERVER_URL}" | tee -a "${MAIN_LOG}"
echo "GPU0=${GPU0}, GPU1=${GPU1}" | tee -a "${MAIN_LOG}"

"${BABYAI_BIN}" --host 127.0.0.1 --port "${SERVER_PORT}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

READY=false
for _ in $(seq 1 60); do
    if curl -fsS "${SERVER_URL}/" >/dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 2
done
if [ "${READY}" != "true" ]; then
    echo "BabyAI server failed to become ready. See ${SERVER_LOG}" | tee -a "${MAIN_LOG}"
    exit 1
fi
echo "BabyAI server ready, pid=${SERVER_PID}, log=${SERVER_LOG}" | tee -a "${MAIN_LOG}"

CUDA_VISIBLE_DEVICES="${GPU0}" "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --stage stage1 \
    --output-root "${OUTPUT_ROOT}" \
    --num-shards 2 \
    --shard-index 0 \
    --stage1-output-path "${STAGE1_DIR}/babyai_sft_step300_oldlogprob_entropy.shard00-of-02.parquet" \
    --progress-every "${STAGE1_BATCH_PROGRESS}" \
    > "${SHARD0_LOG}" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --stage stage1 \
    --output-root "${OUTPUT_ROOT}" \
    --num-shards 2 \
    --shard-index 1 \
    --stage1-output-path "${STAGE1_DIR}/babyai_sft_step300_oldlogprob_entropy.shard01-of-02.parquet" \
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
    --datasets "${DATASETS}" \
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

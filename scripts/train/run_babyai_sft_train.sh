#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SFT_ENV=babyai
# Explicitly invoke bash because OSS/PVC mounts may be noexec.
exec bash "${SCRIPT_DIR}/run_agentgym_sft_train.sh" "$@"

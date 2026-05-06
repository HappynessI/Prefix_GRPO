#!/bin/bash
#
# SciWorld runtime dependency offline wheel packaging script (Python 3.12).
#
# Run this on a network-enabled Linux development machine. It downloads the
# SciWorld environment service dependencies and their full transitive dependency
# tree into third_party/wheels_sciworld/ for H200-side --no-index installs.
#
# H200 training images use Python 3.12, so this script intentionally downloads
# wheels with a Python 3.12 interpreter instead of reusing Python 3.8 wheels.
#
# Usage:
#   bash scripts/prepare_sciworld_wheels_py312.sh
#
# This script is safe to rerun; it replaces old wheels in wheels_sciworld/.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEEL_DIR="${PROJECT_ROOT}/third_party/wheels_sciworld"
REQS_FILE="${PROJECT_ROOT}/third_party/requirements_sciworld_runtime.txt"

# Python 3.12 interpreter provided by the local py312_wheels conda env.
PYTHON312="${PYTHON312:-/home/wyh/miniconda3/envs/py312_wheels/bin/python3.12}"

echo "============================================"
echo "SciWorld offline wheel packaging (Python 3.12)"
echo "============================================"
echo "wheel dir: ${WHEEL_DIR}"
echo "requirements: ${REQS_FILE}"
echo "Python interpreter: ${PYTHON312}"

if [ ! -x "${PYTHON312}" ]; then
    echo "Error: Python 3.12 interpreter not found or not executable: ${PYTHON312}"
    echo "Create it first, for example: conda create -n py312_wheels python=3.12 pip -y"
    exit 1
fi

if [ ! -f "${REQS_FILE}" ]; then
    echo "Error: requirements file does not exist: ${REQS_FILE}"
    exit 1
fi

mkdir -p "${WHEEL_DIR}"
echo "wheel dir ready: ${WHEEL_DIR}"

echo "Cleaning old wheel/source artifacts..."
find "${WHEEL_DIR}" -maxdepth 1 \( -name "*.whl" -o -name "*.tar.gz" \) -type f -delete

echo ""
echo "Downloading SciWorld dependencies and transitive dependencies for Python 3.12..."
"${PYTHON312}" -m pip download \
    -r "${REQS_FILE}" \
    -d "${WHEEL_DIR}"

echo ""
echo "Downloading pip/setuptools/wheel..."
"${PYTHON312}" -m pip download \
    pip setuptools wheel \
    -d "${WHEEL_DIR}"

echo ""
echo "Verifying offline resolution with Python 3.12..."
"${PYTHON312}" -m pip install \
    --dry-run \
    --ignore-installed \
    --no-index \
    --find-links="${WHEEL_DIR}" \
    pip setuptools wheel \
    -r "${REQS_FILE}"

WHEEL_COUNT=$(find "${WHEEL_DIR}" -maxdepth 1 -name "*.whl" -type f | wc -l)
CP38_COUNT=$(find "${WHEEL_DIR}" -maxdepth 1 -name "*cp38*.whl" -type f | wc -l)

echo ""
echo "============================================"
echo "SciWorld wheel packaging complete (Python 3.12)"
echo "============================================"
echo "wheel dir: ${WHEEL_DIR}"
echo "wheel count: ${WHEEL_COUNT}"
echo "cp38 wheel count: ${CP38_COUNT}"
echo ""

if [ "${WHEEL_COUNT}" -eq 0 ]; then
    echo "Error: no .whl files found in ${WHEEL_DIR}"
    exit 1
fi

if [ "${CP38_COUNT}" -ne 0 ]; then
    echo "Error: found cp38 wheels in ${WHEEL_DIR}; this package is not cleanly Python 3.12 aligned."
    find "${WHEEL_DIR}" -maxdepth 1 -name "*cp38*.whl" -type f -print
    exit 1
fi

echo "First 20 wheel files:"
ls -lh "${WHEEL_DIR}"/*.whl 2>/dev/null | head -20
echo ""
echo "Next steps:"
echo "  1. Upload the whole h200_grpo directory, including third_party/wheels_sciworld/."
echo "  2. Also upload third_party/jre_sciworld/; SciWorld requires java at runtime."
echo "  3. On H200, run the SciWorld train script and confirm service ready plus first /reset."
echo "============================================"

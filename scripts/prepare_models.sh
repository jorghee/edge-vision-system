#!/bin/bash
# Prepares (downloads and exports) the required models for the Edge Vision System.
# Models are exported to NCNN format for optimal performance on ARM devices.

set -euo pipefail

# Calculate paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DETECTOR_DIR="${PROJECT_ROOT}/services/detector"
MODELS_DIR="${DETECTOR_DIR}/models"
SCRIPTS_DIR="${DETECTOR_DIR}/scripts"

echo "[1/4] Preparing environment..."
mkdir -p "${MODELS_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is not installed or not in PATH."
    exit 1
fi

if ! python3 -c "import ultralytics" >/dev/null 2>&1; then
    echo "[ERROR] ultralytics package is missing. Please install dependencies first."
    exit 1
fi

echo "[2/4] Downloading models..."
export MODELS_DIR
python3 "${SCRIPTS_DIR}/download_model.py"

BASE_MODEL="${MODELS_DIR}/yolov8n.pt"
PPE_MODEL="${MODELS_DIR}/ppe_detector.pt"

if [ ! -f "${BASE_MODEL}" ]; then
    echo "[ERROR] Failed to download base model: ${BASE_MODEL}"
    exit 1
fi

echo "[3/4] Exporting models to NCNN..."
# The export_model.py script generates the NCNN model directory next to the input model.
export_cmd="python3 \"${SCRIPTS_DIR}/export_model.py\" --base \"${BASE_MODEL}\" --format ncnn"

if [ -f "${PPE_MODEL}" ]; then
    export_cmd="${export_cmd} --ppe \"${PPE_MODEL}\""
fi

eval "${export_cmd}"

echo "[4/4] Verifying exports..."
BASE_NCNN_DIR="${MODELS_DIR}/yolov8n_ncnn_model"
PPE_NCNN_DIR="${MODELS_DIR}/ppe_detector_ncnn_model"

if [ ! -d "${BASE_NCNN_DIR}" ]; then
    echo "[ERROR] NCNN export for base model failed. Directory not found: ${BASE_NCNN_DIR}"
    exit 1
fi

if [ -f "${PPE_MODEL}" ] && [ ! -d "${PPE_NCNN_DIR}" ]; then
    echo "[ERROR] NCNN export for PPE model failed. Directory not found: ${PPE_NCNN_DIR}"
    exit 1
fi

echo "[OK] Models successfully prepared and exported to NCNN format in ${MODELS_DIR}"

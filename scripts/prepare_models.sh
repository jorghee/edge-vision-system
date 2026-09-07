#!/bin/bash
# Prepares (downloads and exports) the required models for the Edge Vision System.
# Models are exported to TFLite format for eKuiper's AI inference pipeline.
# NCNN export is also performed as a secondary format.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DETECTOR_DIR="${PROJECT_ROOT}/services/detector"
MODELS_DIR="${DETECTOR_DIR}/models"
SCRIPTS_DIR="${DETECTOR_DIR}/scripts"

echo "[1/5] Preparing environment..."
mkdir -p "${MODELS_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is not installed or not in PATH."
    exit 1
fi

VENV_DIR="${DETECTOR_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

echo "[INFO] Activating virtual environment and installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install -r "${DETECTOR_DIR}/requirements.txt" --quiet

if ! python3 -c "import ultralytics" >/dev/null 2>&1; then
    echo "[ERROR] Failed to install required packages (e.g., ultralytics)."
    exit 1
fi

echo "[2/5] Downloading models..."
export MODELS_DIR
python3 "${SCRIPTS_DIR}/download_model.py"

BASE_MODEL="${MODELS_DIR}/yolov8n.pt"
PPE_MODEL="${MODELS_DIR}/ppe_detector.pt"

if [ ! -f "${BASE_MODEL}" ]; then
    echo "[ERROR] Failed to download base model: ${BASE_MODEL}"
    exit 1
fi

echo "[3/5] Exporting models to TFLite (primary format for eKuiper)..."
BASE_TFLITE="${MODELS_DIR}/yolov8n_saved_model"
PPE_TFLITE="${MODELS_DIR}/ppe_detector_saved_model"

NEEDS_TFLITE=false
if [ ! -d "${BASE_TFLITE}" ]; then
    NEEDS_TFLITE=true
fi
if [ -f "${PPE_MODEL}" ] && [ ! -d "${PPE_TFLITE}" ]; then
    NEEDS_TFLITE=true
fi

if [ "${NEEDS_TFLITE}" = true ]; then
    export_args="--base \"${BASE_MODEL}\" --format tflite"
    if [ -f "${PPE_MODEL}" ]; then
        export_args="${export_args} --ppe \"${PPE_MODEL}\""
    fi
    eval "python3 \"${SCRIPTS_DIR}/export_model.py\" ${export_args}"
else
    echo "[INFO] TFLite models already exist, skipping export."
fi

echo "[4/5] Exporting models to NCNN (secondary format)..."
BASE_NCNN_DIR="${MODELS_DIR}/yolov8n_ncnn_model"
PPE_NCNN_DIR="${MODELS_DIR}/ppe_detector_ncnn_model"

NEEDS_NCNN=false
if [ ! -d "${BASE_NCNN_DIR}" ]; then
    NEEDS_NCNN=true
fi
if [ -f "${PPE_MODEL}" ] && [ ! -d "${PPE_NCNN_DIR}" ]; then
    NEEDS_NCNN=true
fi

if [ "${NEEDS_NCNN}" = true ]; then
    export_args="--base \"${BASE_MODEL}\" --format ncnn"
    if [ -f "${PPE_MODEL}" ]; then
        export_args="${export_args} --ppe \"${PPE_MODEL}\""
    fi
    eval "python3 \"${SCRIPTS_DIR}/export_model.py\" ${export_args}"
else
    echo "[INFO] NCNN models already exist, skipping export."
fi

echo "[5/5] Verifying exports..."
if [ ! -d "${BASE_TFLITE}" ]; then
    echo "[WARN] TFLite export for base model not found: ${BASE_TFLITE}"
fi
if [ ! -d "${BASE_NCNN_DIR}" ]; then
    echo "[WARN] NCNN export for base model not found: ${BASE_NCNN_DIR}"
fi

echo "[OK] Models prepared. TFLite (eKuiper) and NCNN (fallback) formats in ${MODELS_DIR}"

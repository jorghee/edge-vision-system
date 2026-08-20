#!/bin/bash
# Runs the detector natively on the Raspberry Pi.
# Expects to be invoked from the project root.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETECTOR_DIR="${PROJECT_ROOT}/services/detector"

export MQTT_BROKER="${MQTT_BROKER:-localhost}"
export MQTT_PORT="${MQTT_PORT:-1883}"
export MQTT_TOPIC="${MQTT_TOPIC:-camera/events}"
export CAMERA_ID="${CAMERA_ID:-cam-rpi-01}"
export CAMERA_BACKEND="${CAMERA_BACKEND:-picamera2}"
export INTERVAL_SEC="${INTERVAL_SEC:-5}"
export CONFIDENCE_THR="${CONFIDENCE_THR:-0.45}"
export CAPTURE_WIDTH="${CAPTURE_WIDTH:-640}"
export CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-480}"

# Model: prefer NCNN, fall back to PyTorch
if [ -d "${DETECTOR_DIR}/models/yolov8n_ncnn_model" ]; then
    export MODEL_PATH="${DETECTOR_DIR}/models/yolov8n_ncnn_model"
    echo "[INFO] Using NCNN model (optimized for ARM)"
else
    export MODEL_PATH="${DETECTOR_DIR}/models/yolov8n.pt"
    echo "[WARN] NCNN model not found, using PyTorch (slower)"
fi

if [ -d "${DETECTOR_DIR}/venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source "${DETECTOR_DIR}/venv/bin/activate"
else
    echo "[ERROR] Virtual environment not found. Create with:"
    echo "  cd ${DETECTOR_DIR}"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements-rpi.txt"
    exit 1
fi

echo "[INFO] Verifying camera..."
if command -v libcamera-hello &> /dev/null; then
    libcamera-hello --list-cameras 2>&1 | head -5
    echo ""
else
    echo "[WARN] libcamera-hello not available, continuing anyway..."
fi

echo "============================================"
echo "  Edge Vision System - Raspberry Pi"
echo "  Backend:    $CAMERA_BACKEND"
echo "  Model:      $MODEL_PATH"
echo "  Broker:     $MQTT_BROKER:$MQTT_PORT"
echo "  Topic:      $MQTT_TOPIC"
echo "  Interval:   ${INTERVAL_SEC}s"
echo "  Resolution: ${CAPTURE_WIDTH}x${CAPTURE_HEIGHT}"
echo "============================================"
echo ""

cd "${DETECTOR_DIR}/src"
python3 detector.py

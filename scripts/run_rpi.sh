#!/bin/bash
# ============================================================================
# Run the detector natively on the Raspberry Pi
#
# Prerequisites:
#   1. Python 3.11+ installed
#   2. Virtual environment created: python3 -m venv venv
#   3. Dependencies installed: pip install -r requirements-rpi.txt
#   4. Models exported to NCNN (see export_model.py)
#   5. Infrastructure Docker Compose active:
#      docker compose -f ../docker-compose.rpi.yml up -d
#
# Usage:
#   chmod +x run_rpi.sh
#   ./run_rpi.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
export MQTT_BROKER="${MQTT_BROKER:-localhost}"
export MQTT_PORT="${MQTT_PORT:-1883}"
export MQTT_TOPIC="${MQTT_TOPIC:-camera/events}"
export CAMERA_ID="${CAMERA_ID:-cam-rpi-01}"
export CAMERA_BACKEND="${CAMERA_BACKEND:-picamera2}"
export INTERVAL_SEC="${INTERVAL_SEC:-5}"
export CONFIDENCE_THR="${CONFIDENCE_THR:-0.45}"
export CAPTURE_WIDTH="${CAPTURE_WIDTH:-640}"
export CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-480}"

# Model: use NCNN if exists, PyTorch as fallback
if [ -d "./models/yolov8n_ncnn_model" ]; then
    export MODEL_PATH="./models/yolov8n_ncnn_model"
    echo "[INFO] Using NCNN model (optimized for ARM)"
else
    export MODEL_PATH="./models/yolov8n.pt"
    echo "[WARN] NCNN model not found, using PyTorch (slower)"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[ERROR] Virtual environment not found. Create with:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements-rpi.txt"
    exit 1
fi

# Verify camera
echo "[INFO] Verifying camera..."
if command -v libcamera-hello &> /dev/null; then
    libcamera-hello --list-cameras 2>&1 | head -5
    echo ""
else
    echo "[WARN] libcamera-hello not available, continuing anyway..."
fi

# Execute
echo "============================================"
echo "  Edge Vision System — Raspberry Pi"
echo "  Backend: $CAMERA_BACKEND"
echo "  Modelo:  $MODEL_PATH"
echo "  Broker:  $MQTT_BROKER:$MQTT_PORT"
echo "  Topic:   $MQTT_TOPIC"
echo "  Interval: ${INTERVAL_SEC}s"
echo "  Resolution: ${CAPTURE_WIDTH}x${CAPTURE_HEIGHT}"
echo "============================================"
echo ""

python3 detector.py

#!/bin/bash
# Launches the Edge Vision System on a Raspberry Pi.
# Infrastructure (MQTT, eKuiper, Action Service) runs in Docker.
# The detector runs natively for direct CSI camera access via Picamera2.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETECTOR_DIR="${PROJECT_ROOT}/services/detector"

cd "$PROJECT_ROOT"

echo "[1/5] Starting infrastructure containers (ARM64)..."
docker compose -f docker-compose.rpi.yml up --build -d

echo "[2/5] Waiting for eKuiper to be ready..."
RETRIES=0
MAX_RETRIES=20
until curl -s http://localhost:9081/streams > /dev/null 2>&1; do
    RETRIES=$((RETRIES + 1))
    if [ $RETRIES -ge $MAX_RETRIES ]; then
        echo "[ERROR] eKuiper did not become ready after ${MAX_RETRIES} attempts."
        exit 1
    fi
    echo "  Waiting for eKuiper... (${RETRIES}/${MAX_RETRIES})"
    sleep 3
done

echo "[3/5] Provisioning eKuiper rules..."
bash "${PROJECT_ROOT}/scripts/setup_ekuiper.sh"

echo "[4/5] Preparing detector environment..."
if [ ! -d "${DETECTOR_DIR}/venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv "${DETECTOR_DIR}/venv"
    source "${DETECTOR_DIR}/venv/bin/activate"
    pip install -r "${DETECTOR_DIR}/requirements-rpi.txt"
else
    source "${DETECTOR_DIR}/venv/bin/activate"
fi

if [ ! -d "${DETECTOR_DIR}/models" ] || [ -z "$(ls -A "${DETECTOR_DIR}/models" 2>/dev/null)" ]; then
    echo "[ERROR] No models found in ${DETECTOR_DIR}/models/"
    echo "  Export NCNN models on your laptop first, then transfer them:"
    echo "    cd services/detector/scripts && python3 export_model.py --base yolov8n.pt"
    echo "    rsync -avz services/detector/models/ pi@<RPI_IP>:~/edge-vision-system/services/detector/models/"
    exit 1
fi

echo "[5/5] Starting detector natively..."
bash "${PROJECT_ROOT}/scripts/run_rpi.sh"

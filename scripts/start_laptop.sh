#!/bin/bash
# Launches the full Edge Vision System locally (laptop with USB webcam).
# All services run inside Docker, including the detector.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/3] Building and starting Docker services..."
docker compose up --build -d

echo "[2/3] Waiting for eKuiper to be ready..."
RETRIES=0
MAX_RETRIES=15
until curl -s http://localhost:9081/streams > /dev/null 2>&1; do
    RETRIES=$((RETRIES + 1))
    if [ $RETRIES -ge $MAX_RETRIES ]; then
        echo "[ERROR] eKuiper did not become ready after ${MAX_RETRIES} attempts."
        exit 1
    fi
    echo "  Waiting for eKuiper... (${RETRIES}/${MAX_RETRIES})"
    sleep 2
done

echo "[3/3] Provisioning eKuiper rules..."
bash "${PROJECT_ROOT}/scripts/setup_ekuiper.sh"

echo ""
echo "System is running. Useful commands:"
echo "  docker ps                                              # check containers"
echo "  docker exec mqtt-broker mosquitto_sub -t 'camera/events' -v   # raw events"
echo "  docker exec mqtt-broker mosquitto_sub -t 'edge/alerts' -v     # filtered alerts"
echo "  docker compose down                                    # stop all"

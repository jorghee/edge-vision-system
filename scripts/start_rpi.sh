#!/bin/bash
# Launches the Edge Vision System on a Raspberry Pi.
# All services run in Docker (eKuiper captures video directly).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/3] Starting all containers (ARM64)..."
docker compose -f docker-compose.rpi.yml up --build -d

echo "[2/3] Waiting for eKuiper to be ready..."
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

echo "[3/3] Provisioning eKuiper rules..."
bash "${PROJECT_ROOT}/scripts/setup_ekuiper.sh"

echo ""
echo "System is running. Useful commands:"
echo "  docker ps                                                    # check containers"
echo "  docker exec mqtt-broker mosquitto_sub -t 'edge/alerts' -v    # alerts"
echo "  docker exec mqtt-broker mosquitto_sub -t 'edge/monitor' -v   # all events"
echo "  docker compose -f docker-compose.rpi.yml down                # stop all"

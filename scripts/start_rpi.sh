#!/bin/bash
# Launches the Edge Vision System on a Raspberry Pi.
# All services run in Docker (eKuiper captures video directly).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${MQTT_SERVER_URL:-}" ]; then
    echo "[ERROR] MQTT_SERVER_URL environment variable is not set."
    echo "This script must be run with MQTT_SERVER_URL pointing to the central server."
    echo "Example: MQTT_SERVER_URL=tcp://192.168.1.100:1883 bash scripts/start_rpi.sh"
    exit 1
fi

echo "[1/3] Starting Edge containers (ARM64)..."
MQTT_SERVER_URL="${MQTT_SERVER_URL}" docker compose -f docker-compose.rpi.yml up --build -d

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
MQTT_SERVER_URL="${MQTT_SERVER_URL}" bash "${PROJECT_ROOT}/scripts/setup_ekuiper.sh"

echo ""
echo "Edge System is running and connected to: ${MQTT_SERVER_URL}"
echo "Useful commands:"
echo "  docker ps                                                    # check containers"
echo "  docker compose -f docker-compose.rpi.yml logs -f             # view logs"
echo "  docker compose -f docker-compose.rpi.yml down                # stop all"
echo ""
echo "Verify alerts on the CENTRAL SERVER by subscribing to MQTT topics 'edge/alerts' and 'edge/monitor'."
